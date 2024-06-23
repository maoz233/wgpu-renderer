export default class Renderer {
  constructor() { }

  run() {
    this.init();
  }

  async init() {
    if (!navigator.gpu) {
      throw Error("WebGPU not supported.");
    }
  }
}