import { BUFFERUSAGE_COPY_DST, BUFFERUSAGE_COPY_SRC, StorageBuffer } from 'playcanvas';

import { type ChunkFieldMap, type Layer } from './layout';

/**
 * A GPU-backed slab of one layer's data for one chunk of gaussians.
 *
 * Chunks are acquired from a {@link ChunkManager}, filled by
 * {@link GaussianSource.read}, used by kernels or writers, and then released
 * back to the manager's pool. The underlying `StorageBuffer` is reused across
 * subsequent acquisitions of the same byte size.
 *
 * `count` is the number of gaussians this chunk's allocation can hold; it
 * matches the source's `chunkSize` for all chunks except the final (short)
 * one. `stride` is the bytes per gaussian, dictated by the layer (and, for
 * `color` and `other`, by the SH band count or extras schema).
 */
interface Chunk {
    /** Which layer this chunk holds. */
    readonly layer: Layer;
    /** Number of gaussians this chunk can hold. */
    readonly count: number;
    /** Bytes per gaussian for this chunk's layer. */
    readonly stride: number;
    /** Field name -> byte offset / component descriptor within the stride. */
    readonly fields: ChunkFieldMap;
    /** GPU storage buffer of size `count * stride`. */
    readonly gpu: StorageBuffer;

    /**
     * Materialize a CPU mirror of the chunk's current GPU contents.
     * The returned buffer reflects the data at the time of the call.
     */
    readBack(): Promise<ArrayBuffer>;

    /**
     * View one named field as a strided typed-array over the most recent
     * `readBack()` result. Throws if `readBack()` hasn't been called yet.
     */
    field(name: string): Float32Array | Uint32Array;

    /**
     * Return this chunk's GPU buffer to its `ChunkManager` pool for reuse.
     * After this call the chunk's `gpu` buffer must not be referenced again.
     */
    release(): void;
}

/**
 * Internal release hook called by {@link ChunkImpl.release}. Implemented by
 * the `ChunkManager` to return the underlying buffer to the pool.
 */
type ReleaseFn = (chunk: ChunkImpl) => void;

class ChunkImpl implements Chunk {
    readonly layer: Layer;
    readonly count: number;
    readonly stride: number;
    readonly fields: ChunkFieldMap;
    readonly gpu: StorageBuffer;

    /** Total byte size of the GPU buffer. May be > count * stride if buffer was repurposed from a larger pool slot. */
    readonly byteSize: number;

    private released = false;
    private readonly onRelease: ReleaseFn;
    private cpuMirror: ArrayBuffer | null = null;

    constructor(options: {
        layer: Layer;
        count: number;
        stride: number;
        fields: ChunkFieldMap;
        gpu: StorageBuffer;
        byteSize: number;
        onRelease: ReleaseFn;
    }) {
        this.layer = options.layer;
        this.count = options.count;
        this.stride = options.stride;
        this.fields = options.fields;
        this.gpu = options.gpu;
        this.byteSize = options.byteSize;
        this.onRelease = options.onRelease;
    }

    async readBack(): Promise<ArrayBuffer> {
        if (this.released) {
            throw new Error('Chunk: cannot readBack a released chunk');
        }
        const used = this.count * this.stride;
        const view = await this.gpu.read(0, used);
        // `read` returns a Uint8Array view by default; copy to a fresh ArrayBuffer
        // so the result owns its bytes (PlayCanvas may reuse its staging buffer).
        const ab = (view.buffer as ArrayBuffer).slice(
            view.byteOffset,
            view.byteOffset + view.byteLength
        );
        this.cpuMirror = ab;
        return ab;
    }

    field(name: string): Float32Array | Uint32Array {
        if (this.cpuMirror === null) {
            throw new Error(`Chunk.field: call readBack() before accessing field '${name}'`);
        }
        const f = this.fields[name];
        if (!f) {
            throw new Error(`Chunk.field: unknown field '${name}' for layer '${this.layer}'`);
        }

        // Build a strided view by copying the field's bytes for each gaussian
        // into a tight array. (A true zero-copy strided view isn't possible
        // with TypedArrays when components doesn't span the full stride.)
        const out = f.type === 'float32'
            ? new Float32Array(this.count * f.components)
            : new Uint32Array(this.count * f.components);
        const dv = new DataView(this.cpuMirror);
        const elementSize = 4;
        for (let i = 0; i < this.count; i++) {
            const recordOffset = i * this.stride + f.byteOffset;
            for (let c = 0; c < f.components; c++) {
                const byteOffset = recordOffset + c * elementSize;
                const dstIndex = i * f.components + c;
                if (f.type === 'float32') {
                    (out as Float32Array)[dstIndex] = dv.getFloat32(byteOffset, true);
                } else {
                    (out as Uint32Array)[dstIndex] = dv.getUint32(byteOffset, true);
                }
            }
        }
        return out;
    }

    release(): void {
        if (this.released) {
            return;
        }
        this.released = true;
        this.cpuMirror = null;
        this.onRelease(this);
    }
}

/** Default GPU buffer usage for chunks: writable from CPU + readable back. */
const CHUNK_BUFFER_USAGE = BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC;

export { type Chunk, type ReleaseFn, ChunkImpl, CHUNK_BUFFER_USAGE };
