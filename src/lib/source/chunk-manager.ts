import { type GraphicsDevice, StorageBuffer } from 'playcanvas';

import { CHUNK_BUFFER_USAGE, type Chunk, ChunkImpl } from './chunk';
import { type ChunkFieldMap, type Layer } from './layout';

/**
 * The byte stride and per-field map for a single layer of a source.
 *
 * Sources publish a `LayerLayout` per available layer in their metadata.
 * Callers pass a layout (along with a gaussian count) to {@link ChunkManager.acquire}
 * to receive a properly-sized {@link Chunk}.
 */
type LayerLayout = {
    readonly stride: number;
    readonly fields: ChunkFieldMap;
};

/**
 * A pool-backed allocator for {@link Chunk}s.
 *
 * Typically one manager exists per `GraphicsDevice`, long-lived, and is
 * threaded through any code that needs to acquire chunks. The pool keys on
 * the byte size of the underlying `StorageBuffer`, so a freed 192 MB color
 * chunk can be reused for any subsequent 192 MB allocation regardless of
 * layer — only the `Chunk` wrapper's metadata changes.
 *
 * Pool growth is bounded by `maxPooledBytes` (default 2 GB). On release, if
 * adding the chunk's buffer would exceed the cap, the buffer is destroyed
 * instead of pooled. Call {@link ChunkManager.trim} to free pooled buffers
 * down to a target.
 */
interface ChunkManager {
    /**
     * Acquire a chunk for the given layer/layout, holding `count` gaussians.
     * Reuses a pooled buffer if one of matching byte size is available;
     * otherwise allocates a new `StorageBuffer`.
     */
    acquire(layer: Layer, layout: LayerLayout, count: number): Chunk;

    /** Total bytes currently held by callers (not in the pool). */
    readonly bytesInUse: number;

    /** Total bytes free-listed and ready to be reused. */
    readonly bytesPooled: number;

    /** Free pooled buffers until `bytesPooled <= targetBytes`. */
    trim(targetBytes: number): void;

    /** Free all pooled buffers and clear the pool. In-use chunks are unaffected. */
    destroy(): void;
}

/**
 * Create a chunk manager bound to a `GraphicsDevice`.
 *
 * @param device           - The PlayCanvas graphics device that backs all chunks.
 * @param options.maxPooledBytes - Cap on bytes held in the free list (default 2 GB).
 */
const createChunkManager = (
    device: GraphicsDevice,
    options?: { maxPooledBytes?: number }
): ChunkManager => {
    const maxPooledBytes = options?.maxPooledBytes ?? 2 * 1024 * 1024 * 1024;

    // Pool: byteSize -> stack of free StorageBuffers (LIFO for hot reuse).
    const pool = new Map<number, StorageBuffer[]>();

    let bytesInUse = 0;
    let bytesPooled = 0;

    const release = (chunk: ChunkImpl): void => {
        bytesInUse -= chunk.byteSize;
        if (bytesPooled + chunk.byteSize > maxPooledBytes) {
            // Pool is full; just destroy the buffer rather than holding it.
            chunk.gpu.destroy();
            return;
        }
        let stack = pool.get(chunk.byteSize);
        if (!stack) {
            stack = [];
            pool.set(chunk.byteSize, stack);
        }
        stack.push(chunk.gpu);
        bytesPooled += chunk.byteSize;
    };

    const acquire = (layer: Layer, layout: LayerLayout, count: number): Chunk => {
        if (count <= 0) {
            throw new Error(`ChunkManager.acquire: count must be > 0 (got ${count})`);
        }
        const byteSize = count * layout.stride;
        if (byteSize <= 0) {
            throw new Error(
                `ChunkManager.acquire: derived byteSize=${byteSize} for layer='${layer}' (count=${count}, stride=${layout.stride})`
            );
        }

        // Try to reuse a pooled buffer of the same byte size.
        let gpu: StorageBuffer | undefined;
        const stack = pool.get(byteSize);
        if (stack && stack.length > 0) {
            gpu = stack.pop();
            bytesPooled -= byteSize;
        } else {
            gpu = new StorageBuffer(device, byteSize, CHUNK_BUFFER_USAGE);
        }
        bytesInUse += byteSize;

        return new ChunkImpl({
            layer,
            count,
            stride: layout.stride,
            fields: layout.fields,
            gpu: gpu!,
            byteSize,
            onRelease: release
        });
    };

    const trim = (targetBytes: number): void => {
        if (targetBytes < 0) targetBytes = 0;
        if (bytesPooled <= targetBytes) return;

        // Walk pool entries, destroying buffers until under target.
        // Order: arbitrary (Map iteration order = insertion order); good enough.
        for (const stack of pool.values()) {
            while (stack.length > 0 && bytesPooled > targetBytes) {
                const buf = stack.pop()!;
                bytesPooled -= buf.byteSize;
                buf.destroy();
            }
            if (bytesPooled <= targetBytes) break;
        }
    };

    const destroy = (): void => {
        for (const stack of pool.values()) {
            for (const buf of stack) buf.destroy();
        }
        pool.clear();
        bytesPooled = 0;
    };

    return {
        acquire,
        trim,
        destroy,
        get bytesInUse() { return bytesInUse; },
        get bytesPooled() { return bytesPooled; }
    };
};

export { type ChunkManager, type LayerLayout, createChunkManager };
