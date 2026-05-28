import { Column, type DataTable } from '../data-table';
import { type Transform } from '../utils';
import { createInMemorySource, InMemorySource } from './in-memory-source';
import {
    type ExtraColumn,
    type Layer,
    type SHBands,
    SH_REST_COUNTS,
    colorStride,
    GEOMETRIC_STRIDE,
    POSITION_STRIDE,
    otherLayout
} from './layout';

/** Standard column names that map directly to the canonical layers. */
const POSITION_COLS = ['x', 'y', 'z'] as const;
const GEOMETRIC_COLS = [
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'scale_0', 'scale_1', 'scale_2',
    'opacity'
] as const;
const COLOR_DC_COLS = ['f_dc_0', 'f_dc_1', 'f_dc_2'] as const;

const standardColumnSet = new Set<string>([
    ...POSITION_COLS,
    ...GEOMETRIC_COLS,
    ...COLOR_DC_COLS
]);

/** Determine the SH band count from the highest `f_rest_*` index present. */
const detectShBands = (dataTable: DataTable): SHBands => {
    let highestRest = -1;
    for (const c of dataTable.columns) {
        const m = c.name.match(/^f_rest_(\d+)$/);
        if (m) {
            const n = parseInt(m[1], 10);
            if (n > highestRest) highestRest = n;
        }
    }
    const count = highestRest + 1;
    if (count === 0) return 0;
    if (count === SH_REST_COUNTS[1]) return 1;
    if (count === SH_REST_COUNTS[2]) return 2;
    if (count === SH_REST_COUNTS[3]) return 3;
    throw new Error(`dataTableToSource: unrecognized f_rest_* count: ${count}`);
};

const detectExtras = (dataTable: DataTable): ExtraColumn[] => {
    const extras: ExtraColumn[] = [];
    for (const c of dataTable.columns) {
        if (standardColumnSet.has(c.name)) continue;
        if (/^f_rest_\d+$/.test(c.name)) continue;
        // Map dtype to one of float32/uint32 for the canonical 'other' layer.
        const type: 'float32' | 'uint32' = (
            c.dataType === 'float32' || c.dataType === 'float64'
        ) ? 'float32' : 'uint32';
        extras.push({ name: c.name, type });
    }
    return extras;
};

/**
 * Split a `Float32Array` of N gaussians × stride floats into per-chunk
 * `ArrayBuffer` slabs of `chunkSize` gaussians each (last chunk may be short).
 */
const splitToChunks = (
    interleaved: Float32Array,
    numGaussians: number,
    floatsPerRow: number,
    chunkSize: number
): ArrayBuffer[] => {
    const out: ArrayBuffer[] = [];
    let rowsRemaining = numGaussians;
    let rowOffset = 0;
    while (rowsRemaining > 0) {
        const rows = Math.min(chunkSize, rowsRemaining);
        const slice = interleaved.subarray(
            rowOffset * floatsPerRow,
            (rowOffset + rows) * floatsPerRow
        );
        // Copy into a fresh ArrayBuffer so the chunk owns its bytes.
        const ab = new ArrayBuffer(rows * floatsPerRow * 4);
        new Float32Array(ab).set(slice);
        out.push(ab);
        rowOffset += rows;
        rowsRemaining -= rows;
    }
    return out;
};

const splitToChunksU32 = (
    interleaved: Uint32Array,
    numGaussians: number,
    u32sPerRow: number,
    chunkSize: number
): ArrayBuffer[] => {
    const out: ArrayBuffer[] = [];
    let rowsRemaining = numGaussians;
    let rowOffset = 0;
    while (rowsRemaining > 0) {
        const rows = Math.min(chunkSize, rowsRemaining);
        const slice = interleaved.subarray(
            rowOffset * u32sPerRow,
            (rowOffset + rows) * u32sPerRow
        );
        const ab = new ArrayBuffer(rows * u32sPerRow * 4);
        new Uint32Array(ab).set(slice);
        out.push(ab);
        rowOffset += rows;
        rowsRemaining -= rows;
    }
    return out;
};

/** Default chunk size: 1M gaussians per chunk. */
const DEFAULT_CHUNK_SIZE = 1 << 20;

/**
 * Convert a legacy `DataTable` into a `GaussianSource` by repacking its
 * columnar data into the canonical per-layer interleaved chunk layout.
 *
 * Detects SH band count from the highest `f_rest_*` index, identifies
 * non-standard columns as `other`-layer extras, and copies each gaussian's
 * fields into the appropriate per-layer chunk buffer.
 *
 * Used during the 3.0 migration by readers that haven't yet been ported to
 * native chunked decoding — they call this at the end of their existing
 * decode to upgrade to the new return type. Eventually each reader can be
 * upgraded to skip the intermediate DataTable entirely.
 */
const dataTableToSource = (
    dataTable: DataTable,
    chunkSize: number = DEFAULT_CHUNK_SIZE
): InMemorySource => {
    const N = dataTable.numRows;
    const shBands = detectShBands(dataTable);
    const numRest = SH_REST_COUNTS[shBands];
    const extras = detectExtras(dataTable);

    const transform: Transform = dataTable.transform;

    // Detect which canonical layers are present.
    const hasPosition = POSITION_COLS.every(c => dataTable.hasColumn(c));
    const hasGeometric = GEOMETRIC_COLS.every(c => dataTable.hasColumn(c));
    const hasColor = COLOR_DC_COLS.every(c => dataTable.hasColumn(c));
    const hasOther = extras.length > 0;

    // Build interleaved arrays per layer.
    const positionBuffers: ArrayBuffer[] | undefined = hasPosition ? (() => {
        const arr = new Float32Array(N * 3);
        const x = dataTable.getColumnByName('x')!.data as Float32Array;
        const y = dataTable.getColumnByName('y')!.data as Float32Array;
        const z = dataTable.getColumnByName('z')!.data as Float32Array;
        for (let i = 0; i < N; i++) {
            arr[i * 3 + 0] = x[i];
            arr[i * 3 + 1] = y[i];
            arr[i * 3 + 2] = z[i];
        }
        return splitToChunks(arr, N, 3, chunkSize);
    })() : undefined;

    const geometricBuffers: ArrayBuffer[] | undefined = hasGeometric ? (() => {
        const arr = new Float32Array(N * 8);
        const r0 = dataTable.getColumnByName('rot_0')!.data as Float32Array;
        const r1 = dataTable.getColumnByName('rot_1')!.data as Float32Array;
        const r2 = dataTable.getColumnByName('rot_2')!.data as Float32Array;
        const r3 = dataTable.getColumnByName('rot_3')!.data as Float32Array;
        const s0 = dataTable.getColumnByName('scale_0')!.data as Float32Array;
        const s1 = dataTable.getColumnByName('scale_1')!.data as Float32Array;
        const s2 = dataTable.getColumnByName('scale_2')!.data as Float32Array;
        const op = dataTable.getColumnByName('opacity')!.data as Float32Array;
        for (let i = 0; i < N; i++) {
            const o = i * 8;
            arr[o + 0] = r0[i];
            arr[o + 1] = r1[i];
            arr[o + 2] = r2[i];
            arr[o + 3] = r3[i];
            arr[o + 4] = s0[i];
            arr[o + 5] = s1[i];
            arr[o + 6] = s2[i];
            arr[o + 7] = op[i];
        }
        return splitToChunks(arr, N, 8, chunkSize);
    })() : undefined;

    const colorBuffers: ArrayBuffer[] | undefined = hasColor ? (() => {
        const floatsPerRow = 3 + numRest;
        const arr = new Float32Array(N * floatsPerRow);
        const dc0 = dataTable.getColumnByName('f_dc_0')!.data as Float32Array;
        const dc1 = dataTable.getColumnByName('f_dc_1')!.data as Float32Array;
        const dc2 = dataTable.getColumnByName('f_dc_2')!.data as Float32Array;
        const restCols: Float32Array[] = [];
        for (let r = 0; r < numRest; r++) {
            restCols.push(dataTable.getColumnByName(`f_rest_${r}`)!.data as Float32Array);
        }
        for (let i = 0; i < N; i++) {
            const o = i * floatsPerRow;
            arr[o + 0] = dc0[i];
            arr[o + 1] = dc1[i];
            arr[o + 2] = dc2[i];
            for (let r = 0; r < numRest; r++) arr[o + 3 + r] = restCols[r][i];
        }
        return splitToChunks(arr, N, floatsPerRow, chunkSize);
    })() : undefined;

    const otherBuffers: ArrayBuffer[] | undefined = hasOther ? (() => {
        const u32s = extras.length;
        const arr = new Uint32Array(N * u32s);
        const f32View = new Float32Array(arr.buffer);
        const cols: Column[] = extras.map(e => dataTable.getColumnByName(e.name)!);
        for (let i = 0; i < N; i++) {
            const o = i * u32s;
            for (let e = 0; e < u32s; e++) {
                if (extras[e].type === 'float32') {
                    f32View[o + e] = cols[e].data[i] as number;
                } else {
                    arr[o + e] = cols[e].data[i] as number;
                }
            }
        }
        return splitToChunksU32(arr, N, u32s, chunkSize);
    })() : undefined;

    void otherLayout; // referenced by createInMemorySource for layout derivation

    return createInMemorySource({
        numGaussians: N,
        chunkSize,
        shBands,
        extraColumns: extras,
        transform,
        position: positionBuffers ? [positionBuffers] : undefined,
        geometric: geometricBuffers ? [geometricBuffers] : undefined,
        color: colorBuffers ? [colorBuffers] : undefined,
        other: otherBuffers ? [otherBuffers] : undefined
    });
};

// Reference the constants once so they appear used to TS for type-check stability.
void POSITION_STRIDE; void GEOMETRIC_STRIDE; void colorStride;
type _Layer = Layer;

export { dataTableToSource, DEFAULT_CHUNK_SIZE };
