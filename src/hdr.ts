export default class HDRLoader {
  private buffer: ArrayBuffer;

  public constructor() {}

  public async load(url: string): Promise<void> {
    this.buffer = await this.loadBuffer(url);
    this.parseHDR();
  }

  private async loadBuffer(url: string): Promise<ArrayBuffer> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch .hdr file: ${url}`);
    }

    const arrayBuffer = await res.arrayBuffer();

    return arrayBuffer;
  }

  private parseHDR(): void {
    const bytes = new Uint8Array(this.buffer);
    let header: string;
    let i = 0;

    while (i < bytes.length && bytes[i] !== 0x0a) {
      header += String.fromCharCode(bytes[i]);
      ++i;
    }
    header += "\n";

    while (
      i < bytes.length - 1 &&
      (bytes[i] !== 0x0a || bytes[i + 1] !== 0x0a)
    ) {
      header += String.fromCharCode(bytes[i]);
      ++i;
    }
    i += 2;
  }
}
