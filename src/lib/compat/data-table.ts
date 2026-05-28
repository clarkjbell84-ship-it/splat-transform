import { Column, DataTable } from '../data-table';
import {
    type ChunkManager,
    type GaussianSource,
    SH_REST_COUNTS,
    createInMemorySource
} from '../source';
import { type Layer } from '../source/layout';

/**
 * Compatibility shim: materialize a `GaussianSource` into the legacy
 * columnar `DataTable` representation.
 *
 * Used during the 3.0 migration so downstream code that hasn't yet been
 * ported (writers, process actions, supersplat) can still consume sources
 * as `DataTable`. Each available layer is read chunk-by-chunk and scattered
 * into the appropriate named columns (`x, y, z, rot_*, scale_*, opacity,
 * f_dc_*, f_rest_*`, plus extras).
 *
 * Once every downstream consumer has been ported to consume `GaussianSource`
 * directly, this shim and the `DataTable` class can be removed.
 */
const materializeToDataTable = async (
    src: GaussianSource,
    manager: ChunkManager
): Promise<DataTable> => {
    const { meta } = src;
    const N = meta.numGaussians;

    const x = new Float32Array(N);
    const y = new Float32Array(N);
    const z = new Float32Array(N);

    const rot0 = new Float32Array(N);
    const rot1 = new Float32Array(N);
    const rot2 = new Float32Array(N);
    const rot3 = new Float32Array(N);
    const scale0 = new Float32Array(N);
    const scale1 = new Float32Array(N);
    const scale2 = new Float32Array(N);
    const opacity = new Float32Array(N);

    const dc0 = new Float32Array(N);
    const dc1 = new Float32Array(N);
    const dc2 = new Float32Array(N);

    const numRest = SH_REST_COUNTS[meta.shBands];
    const restArrays: Float32Array[] = [];
    for (let i = 0; i < numRest; i++) restArrays.push(new Float32Array(N));

    const extraArrays: { name: string; type: 'float32' | 'uint32'; data: Float32Array | Uint32Array }[] =
        meta.extraColumns.map(e => ({
            name: e.name,
            type: e.type,
            data: e.type === 'float32' ? new Float32Array(N) : new Uint32Array(N)
        }));

    const wantsPosition = meta.availableLayers.has('position');
    const wantsGeometric = meta.availableLayers.has('geometric');
    const wantsColor = meta.availableLayers.has('color');
    const wantsOther = meta.availableLayers.has('other') && extraArrays.length > 0;

    const chunkSize = meta.chunkSize;
    const numChunks = meta.numChunks[0];

    for (let k = 0; k < numChunks; k++) {
        const rowStart = k * chunkSize;
        const count = Math.min(chunkSize, N - rowStart);

        const layouts = meta.layouts;
        const chunks: { layer: Layer; chunk: import('../source').Chunk }[] = [];
        const req: { chunkIndex: number; lod: number; position?: import('../source').Chunk; geometric?: import('../source').Chunk; color?: import('../source').Chunk; other?: import('../source').Chunk } = { chunkIndex: k, lod: 0 };

        if (wantsPosition) {
            const c = manager.acquire('position', layouts.position!, count);
            req.position = c;
            chunks.push({ layer: 'position', chunk: c });
        }
        if (wantsGeometric) {
            const c = manager.acquire('geometric', layouts.geometric!, count);
            req.geometric = c;
            chunks.push({ layer: 'geometric', chunk: c });
        }
        if (wantsColor) {
            const c = manager.acquire('color', layouts.color!, count);
            req.color = c;
            chunks.push({ layer: 'color', chunk: c });
        }
        if (wantsOther) {
            const c = manager.acquire('other', layouts.other!, count);
            req.other = c;
            chunks.push({ layer: 'other', chunk: c });
        }

        await src.read(req);

        // For each layer, read back to CPU and scatter into the legacy column arrays.
        for (const { layer, chunk } of chunks) {
            const ab = await chunk.readBack();
            const f32 = new Float32Array(ab);

            if (layer === 'position') {
                // stride 12 bytes = 3 f32; row i -> [x,y,z] starting at i*3
                for (let i = 0; i < count; i++) {
                    const di = rowStart + i;
                    const si = i * 3;
                    x[di] = f32[si + 0];
                    y[di] = f32[si + 1];
                    z[di] = f32[si + 2];
                }
            } else if (layer === 'geometric') {
                // stride 32 bytes = 8 f32; [rotW, rotX, rotY, rotZ, sX, sY, sZ, opacity]
                for (let i = 0; i < count; i++) {
                    const di = rowStart + i;
                    const si = i * 8;
                    rot0[di] = f32[si + 0];
                    rot1[di] = f32[si + 1];
                    rot2[di] = f32[si + 2];
                    rot3[di] = f32[si + 3];
                    scale0[di] = f32[si + 4];
                    scale1[di] = f32[si + 5];
                    scale2[di] = f32[si + 6];
                    opacity[di] = f32[si + 7];
                }
            } else if (layer === 'color') {
                // stride = 12 + 4 * numRest bytes; layout [dc0, dc1, dc2, rest_0, rest_1, ...]
                const stride = 3 + numRest;
                for (let i = 0; i < count; i++) {
                    const di = rowStart + i;
                    const si = i * stride;
                    dc0[di] = f32[si + 0];
                    dc1[di] = f32[si + 1];
                    dc2[di] = f32[si + 2];
                    for (let r = 0; r < numRest; r++) {
                        restArrays[r][di] = f32[si + 3 + r];
                    }
                }
            } else { // 'other'
                // Each extra column is 1 element (4 bytes) per gaussian in declared order.
                const u32 = new Uint32Array(ab);
                for (let i = 0; i < count; i++) {
                    const di = rowStart + i;
                    for (let e = 0; e < extraArrays.length; e++) {
                        if (extraArrays[e].type === 'float32') {
                            (extraArrays[e].data as Float32Array)[di] = f32[i * extraArrays.length + e];
                        } else {
                            (extraArrays[e].data as Uint32Array)[di] = u32[i * extraArrays.length + e];
                        }
                    }
                }
            }
        }

        for (const { chunk } of chunks) chunk.release();
    }

    const columns: Column[] = [];
    if (wantsPosition) {
        columns.push(new Column('x', x), new Column('y', y), new Column('z', z));
    }
    if (wantsGeometric) {
        columns.push(
            new Column('rot_0', rot0),
            new Column('rot_1', rot1),
            new Column('rot_2', rot2),
            new Column('rot_3', rot3),
            new Column('scale_0', scale0),
            new Column('scale_1', scale1),
            new Column('scale_2', scale2),
            new Column('opacity', opacity)
        );
    }
    if (wantsColor) {
        columns.push(
            new Column('f_dc_0', dc0),
            new Column('f_dc_1', dc1),
            new Column('f_dc_2', dc2)
        );
        for (let r = 0; r < numRest; r++) {
            columns.push(new Column(`f_rest_${r}`, restArrays[r]));
        }
    }
    if (wantsOther) {
        for (const e of extraArrays) {
            columns.push(new Column(e.name, e.data));
        }
    }

    return new DataTable(columns, meta.transform);
};

// Silence unused warning until first caller; createInMemorySource is re-exported
// for future use by the SPZ/MJS paths.
void createInMemorySource;

export { materializeToDataTable };
