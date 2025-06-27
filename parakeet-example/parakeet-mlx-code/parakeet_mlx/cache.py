"""Cache mechanisms for the Conformer model to support efficient streaming inference."""

import mlx.core as mx


class ConformerCache:
    """A cache for Conformer model layers.

    This cache stores key-value pairs for attention layers and context for
    convolutional layers. It grows dynamically as more data is processed.

    Attributes:
        keys (mx.array | None): Cached keys for the attention layer.
        values (mx.array | None): Cached values for the attention layer.
        conv (mx.array | None): Cached context for the convolutional layer.
        offset (int): The current offset in the key-value cache.
        step (int): The allocation step size for the cache.
    """

    keys: mx.array | None
    values: mx.array | None
    conv: mx.array | None

    offset: int
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.conv = None
        self.offset = 0

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Update the key-value cache and return the full sequence.

        Args:
            keys (mx.array): The new keys to add to the cache.
                Shape is `(batch, head, seq, dim)`.
            values (mx.array): The new values to add to the cache.
                Shape is `(batch, head, seq, dim)`.

        Returns:
            A tuple containing the updated keys and values, including all
            historical data.
        """
        prev = self.offset
        if (
            self.keys is None
            or self.values is None
            or (prev + keys.shape[2]) > self.keys.shape[2]
        ):
            B, H, S, D_KEYS = keys.shape
            _, _, _, D_VALUES = values.shape
            S_CACHE = ((self.step + S - 1) // self.step) * self.step

            new_k = mx.zeros((B, H, S_CACHE, D_KEYS), keys.dtype)
            new_v = mx.zeros((B, H, S_CACHE, D_VALUES), keys.dtype)

            if self.keys is None or self.values is None:  # type safety!
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def update_and_fetch_conv(self, x: mx.array, padding: int = 0) -> mx.array:
        """Update the convolution cache and return the input with cached context.

        Args:
            x (mx.array): The input array of shape `(B, S, D)`.
            padding (int): The required left padding for the convolution.
                Defaults to 0.

        Returns:
            mx.array: The input `x` concatenated with the cached left context,
                with shape `(B, padding + S, D)`, plus right padding.
        """
        if padding == 0:
            return x

        B, S, D = x.shape

        if self.conv is None:
            self.conv = mx.zeros((B, padding, D), x.dtype)

        tokens_to_cache = min(padding, S)

        cache_update = x[:, S - tokens_to_cache : S, :]

        if tokens_to_cache < padding:
            self.conv = mx.concatenate(
                [self.conv[:, tokens_to_cache:, :], cache_update], axis=1
            )
        else:
            self.conv = cache_update

        result = mx.concatenate([self.conv, x], axis=1)
        result = mx.pad(result, ((0, 0), (0, padding), (0, 0)))

        return result


class RotatingConformerCache(ConformerCache):
    """A fixed-size cache for Conformer layers that uses a ring buffer.

    This cache is designed for streaming inference where only a finite
    amount of context is kept. When the cache is full, older entries
    are overwritten.

    Attributes:
        capacity (int): The maximum number of feature frames to store.
        cache_drop_size (int): The number of frames to drop from the
            start of the new chunk before caching.
    """

    capacity: int
    cache_drop_size: int

    def __init__(self, capacity: int, cache_drop_size: int = 0):
        super().__init__()

        self.capacity = capacity
        self.cache_drop_size = cache_drop_size

    def _ring_append(self, buf: mx.array, new: mx.array):
        """Append data to the ring buffer.

        Args:
            buf (mx.array): The ring buffer.
            new (mx.array): The new data to append.
        """
        C = self.capacity
        pos = self.offset % C
        T = new.shape[2]
        first = min(T, C - pos)
        buf[..., pos : pos + first, :] = new[..., :first, :]
        if first < T:
            buf[..., : T - first, :] = new[..., first:, :]

    def update_and_fetch_kv(self, keys: mx.array, values: mx.array):
        """Update the rotating key-value cache and return the combined sequence.

        Args:
            keys (mx.array): The new keys to add to the cache.
                Shape is `(batch, head, seq, dim)`.
            values (mx.array): The new values to add to the cache.
                Shape is `(batch, head, seq, dim)`.

        Returns:
            A tuple containing the cached history combined with the new keys
            and values.
        """
        B, H, S, D = keys.shape

        if self.keys is None or self.values is None:
            self.keys = mx.zeros((B, H, self.capacity, D), keys.dtype)
            self.values = mx.zeros((B, H, self.capacity, D), keys.dtype)

        if self.offset < self.capacity:
            hist_k = self.keys[..., : self.offset, :]
            hist_v = self.values[..., : self.offset, :]
        else:
            shift = -(self.offset % self.capacity)
            hist_k = mx.roll(self.keys, shift, 2)
            hist_v = mx.roll(self.values, shift, 2)

        k_out = mx.concatenate([hist_k, keys], axis=2)
        v_out = mx.concatenate([hist_v, values], axis=2)

        drop = self.cache_drop_size
        to_cache = min(max(0, S - drop), self.capacity)

        if to_cache > 0:
            k_chunk = keys[
                ...,
                S - self.cache_drop_size - to_cache : S - self.cache_drop_size,
                :,
            ]
            v_chunk = values[
                ...,
                S - self.cache_drop_size - to_cache : S - self.cache_drop_size,
                :,
            ]
            self._ring_append(self.keys, k_chunk)
            self._ring_append(self.values, v_chunk)
            self.offset += to_cache

        return k_out, v_out

    def update_and_fetch_conv(self, x: mx.array, padding: int = 0) -> mx.array:
        """Update the rotating convolution cache and return the input with context.

        Args:
            x (mx.array): The input array of shape `(B, S, D)`.
            padding (int): The required left padding for the convolution.
                Defaults to 0.

        Returns:
            mx.array: The input `x` concatenated with the cached left context,
            plus right padding.
        """
        if padding == 0:
            return x

        B, S, D = x.shape

        if self.conv is None:
            self.conv = mx.zeros((B, padding, D), x.dtype)

        if self.cache_drop_size < S:
            tokens_to_cache = min(padding, S - self.cache_drop_size)
            cache_update = x[:, S - tokens_to_cache : S, :]

            if tokens_to_cache < padding:
                self.conv = mx.concatenate(
                    [self.conv[:, tokens_to_cache:, :], cache_update], axis=1
                )
            else:
                self.conv = cache_update

        result = mx.concatenate([self.conv, x], axis=1)
        result = mx.pad(result, ((0, 0), (0, padding), (0, 0)))

        return result
