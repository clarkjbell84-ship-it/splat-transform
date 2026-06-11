import { parentPort } from 'node:worker_threads';

import { Column, DataTable, type TypedArray } from '../lib/data-table/data-table';
import { quantize1d } from '../lib/spatial/quantize-1d';
import { WebPCodec } from '../lib/utils/webp-codec';

/**
 * Worker thread entry point for the SOG writer's CPU-heavy steps. Receives
 * one task at a time from SogWorkerPool and posts the result back, with
 * typed array buffers transferred rather than copied. Built as a separate
 * bundle (dist/sog-worker.mjs) alongside the CLI.
 */

type WorkerMessage = {
    type: 'quantize1d';
    columns: { name: string, data: TypedArray }[];
    k?: number;
    alpha?: number;
} | {
    type: 'encodeWebp';
    rgba: Uint8Array;
    width: number;
    height: number;
};

let codec: Promise<WebPCodec>;

parentPort.on('message', async (message: WorkerMessage) => {
    try {
        if (message.type === 'encodeWebp') {
            codec = codec ?? WebPCodec.create();
            const webp = (await codec).encodeLosslessRGBA(message.rgba, message.width, message.height);
            parentPort.postMessage({ result: webp }, [webp.buffer as ArrayBuffer]);
        } else {
            const { centroids, labels } = quantize1d(
                new DataTable(message.columns.map(c => new Column(c.name, c.data))),
                message.k,
                message.alpha
            );
            const centroidsData = centroids.getColumn(0).data;
            const labelColumns = labels.columns.map(c => ({ name: c.name, data: c.data }));
            parentPort.postMessage({
                result: { centroids: centroidsData, labels: labelColumns }
            }, [
                centroidsData.buffer as ArrayBuffer,
                ...labelColumns.map(c => c.data.buffer as ArrayBuffer)
            ]);
        }
    } catch (err) {
        parentPort.postMessage({ error: err instanceof Error ? err.message : String(err) });
    }
});
