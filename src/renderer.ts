export default class Renderer {
  constructor() { }

  async init() {
    if (!navigator.gpu) {
      throw Error("WebGPU not supported.");
    }
  }
}