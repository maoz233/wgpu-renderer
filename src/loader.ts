import { glTF } from "@/types/glTF";

export default class glTFLoader {
  private gltf: glTF;

  public constructor() {}

  public async load(url: string): Promise<void> {
    const res = await fetch(url);
    this.gltf = (await res.json()) as glTF;

    if ("2.0" !== this.gltf.asset.version) {
      throw new Error(`Unsupported glTF version: ${this.gltf.asset.version}`);
    }
  }
}
