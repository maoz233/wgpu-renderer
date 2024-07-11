export function rand(min?: number, max?: number) {
  if (undefined === min) {
    min = 0;
    max = 1;
  } else if (undefined === max) {
    max = min;
    min = 0;
  }

  return min + Math.random() * (max - min);
}

export async function loadImageBitmap(url: string) {
  const res = await fetch(url);
  const blob = await res.blob();

  return await createImageBitmap(blob, { colorSpaceConversion: "none" });
}

export function calculateMipLevelCount(...sizes: number[]) {
  const maxSize = Math.max(...sizes);

  return (1 + Math.log2(maxSize)) | 0;
}

export class RollingAverage {
  private total: number;
  private samples: number[];
  private cursor: number;
  private sampleCount: number;

  public constructor(sampleCount = 100) {
    this.total = 0;
    this.samples = [];
    this.cursor = 0;
    this.sampleCount = sampleCount;
  }

  public set value(value: number) {
    this.total += value - (this.samples[this.cursor] || 0);
    this.samples[this.cursor] = value;
    this.cursor = (this.cursor + 1) % this.sampleCount;
  }

  public get value() {
    return this.total / this.samples.length;
  }
}
