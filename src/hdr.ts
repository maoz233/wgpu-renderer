export default class HDRLoader {
  private buffer: ArrayBuffer;

  public constructor() {}

  public async load(url: string): Promise<void> {
    this.buffer = await this.loadBuffer(url);
  }

  private async loadBuffer(url: string): Promise<ArrayBuffer> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch .hdr file: ${url}`);
    }

    const arrayBuffer = await res.arrayBuffer();

    return arrayBuffer;
  }
}
