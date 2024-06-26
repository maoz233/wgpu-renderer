import shaderCode from "./shaders/triangle.wgsl";

const vertices = new Float32Array([0.0, 0.6, 0, 1, 1, 0, 0, 1, -0.5, -0.6, 0, 1, 0, 1, 0, 1, 0.5, -0.6, 0, 1, 0, 0, 1, 1,]);

export default class Renderer {
  private adapter: GPUAdapter;
  private device: GPUDevice;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;
  private vertexBuffer: GPUBuffer;
  private renderPipeline: GPURenderPipeline;

  public constructor() { }

  public async render() {
    await this.init();
    this.run();
  }

  private async init() {
    this.checkWebGPUSupport();
    await this.requestAdapter();
    await this.requestDevice();
    this.getCanvas();
    this.configContext();
    this.createVertexBuffer();
    this.createRenderPipeline();
  }

  private checkWebGPUSupport() {
    if (!navigator.gpu) {
      throw Error("WebGPU not supported.");
    }
  }

  private async requestAdapter() {
    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw Error("Failed to request WebGPU adapter.");
    }
  }

  private async requestDevice() {
    this.device = await this.adapter.requestDevice();
    if (!this.device) {
      throw Error("Failed to request WebGPU device.");
    }
  }

  private getCanvas() {
    this.canvas = document.querySelector('canvas');
    if (!this.canvas) {
      throw Error("Failed to find canvas element.");
    }
  }

  private configContext() {
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) {
      throw Error("Failed to get WebGPU context from canvas.");
    }

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "premultiplied",
    });
  }

  private createVertexBuffer() {
    this.vertexBuffer = this.createBuffer("Vertex Buffer", vertices, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
  }

  private createRenderPipeline() {
    const shaderModule = this.createShaderModule("Triangle Shader Module", shaderCode);

    const attributes: GPUVertexAttribute[] = [
      {
        shaderLocation: 0,
        offset: 0,
        format: "float32x4",
      },
      {
        shaderLocation: 1,
        offset: 16,
        format: "float32x4",
      },
    ];

    const vertexBuffers: GPUVertexBufferLayout[] = [
      {
        attributes,
        arrayStride: 32,
        stepMode: "vertex",
      },
    ];

    const pipelineDescriptor: GPURenderPipelineDescriptor = {
      vertex: {
        module: shaderModule,
        entryPoint: "vertex_main",
        buffers: vertexBuffers,
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fragment_main",
        targets: [
          {
            format: this.presentationFormat,
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
      },
      layout: "auto",
    };

    this.renderPipeline = this.device.createRenderPipeline(pipelineDescriptor);
  }

  private createBuffer(label: string, data: Float32Array, usage: GPUBufferUsageFlags) {
    const buffer = this.device.createBuffer({
      label,
      size: data.byteLength,
      usage,
    });
    this.device.queue.writeBuffer(buffer, 0, data, 0, data.length);

    return buffer;
  }

  private createShaderModule(label: string, code: string) {
    const shaderModule = this.device.createShaderModule({
      label,
      code,
    })

    return shaderModule;
  }

  private run() {
    requestAnimationFrame(() => this.drawFrame());
  }

  private drawFrame() {
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
    const colorAttachments: GPURenderPassColorAttachment[] = [
      {
        view: textureView,
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ];

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments,
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipeline);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.draw(3);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(() => this.drawFrame());
  }
}
