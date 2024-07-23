export default class HDRLoader {
  private buffer: ArrayBuffer;
  private width_: number;
  private height_: number;
  private data_: Float32Array;

  public constructor() {}

  public get width(): number {
    return this.width_;
  }

  public get height(): number {
    return this.height_;
  }

  public get bytes(): Float32Array {
    return this.data_;
  }

  public async load(url: string): Promise<void> {
    await this.loadBuffer(url);
    this.parseHDR();
  }

  private async loadBuffer(url: string): Promise<void> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch .hdr file: ${url}`);
    }

    this.buffer = await res.arrayBuffer();
  }

  private parseHDR(): void {
    const bytes = new Uint8Array(this.buffer);

    // HDR header
    const decoder = new TextDecoder("ascii");
    let index = 0;
    let header = "";

    while (index < bytes.length) {
      const char = decoder.decode(bytes.slice(index, index + 1));
      header += char;
      index++;
      if (char === "\n" && header.includes("-Y ") && header.includes(" +X ")) {
        break;
      }
    }

    if (
      !header.includes("32-bit_rle_rgbe") &&
      !header.includes("32-bit_rle_xyze")
    ) {
      throw new Error("HDRLoader: Unsupported .hdr format.");
    }

    const resolution = header.match(/-Y (\d+) \+X (\d+)/);
    if (!resolution) {
      throw new Error("HDRLoader: No resolution found.");
    }

    this.width_ = parseInt(resolution[2]);
    this.height_ = parseInt(resolution[1]);

    // HDR pixel bytes
    this.data_ = new Float32Array(this.width_ * this.height_ * 3);
    let currentPixel = 0;
    while (index < bytes.length) {
      const scanlineHeader = bytes.slice(index, index + 4);
      index += 4;

      if (
        scanlineHeader[0] !== 2 ||
        scanlineHeader[1] !== 2 ||
        (scanlineHeader[2] << 8) + scanlineHeader[3] !== this.width_
      ) {
        throw new Error("HDRLoader: Only support RLE format.");
      }

      const scanline = new Uint8Array(this.width_ * 4);
      for (let i = 0; i < 4; i++) {
        let position = 0;
        while (position < this.width_) {
          const count = bytes[index++];
          if (count > 128) {
            const runLength = count - 128;
            const value = bytes[index++];
            for (let j = 0; j < runLength; j++) {
              scanline[i + position * 4] = value;
              position++;
            }
          } else {
            for (let j = 0; j < count; j++) {
              scanline[i + position * 4] = bytes[index++];
              position++;
            }
          }
        }
      }

      for (let x = 0; x < this.width_; x++) {
        const r = scanline[x * 4];
        const g = scanline[x * 4 + 1];
        const b = scanline[x * 4 + 2];
        const e = scanline[x * 4 + 3];
        const scale = Math.pow(2, e - 128) / 255;

        this.data_[currentPixel++] = r * scale;
        this.data_[currentPixel++] = g * scale;
        this.data_[currentPixel++] = b * scale;
      }
    }
  }
}
