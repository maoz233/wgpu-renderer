import shaderCode from "@/shaders/triangle.wgsl";

const vertices = new Float32Array([
  0.0, 0.6, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -0.5, -0.6, 0.0, 1.0, 0.0, 1.0, 0.0,
  1.0, 0.5, -0.6, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
]);

export default class Renderer {
  private adapter: GPUAdapter;
  private device: GPUDevice;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;
  private vertexBuffer: GPUBuffer;
  private uniformValues: Float32Array;
  private scaleOffset: number;
  private offsetOffset: number;
  private uniformBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup;
  private renderPipeline: GPURenderPipeline;

  public constructor() {
    this.drawFrame = this.drawFrame.bind(this);
  }

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
    this.createRenderPipeline();
    this.createVertexBuffer();
    this.createUniformBuffer();
  }

  private checkWebGPUSupport() {
    if (!navigator.gpu) {
      throw Error("WebGPU not supported.");
    }
  }

  private async requestAdapter() {
    this.adapter = await navigator.gpu?.requestAdapter();
    if (!this.adapter) {
      throw Error("Failed to request WebGPU adapter.");
    }
  }

  private async requestDevice() {
    this.device = await this.adapter?.requestDevice();
    if (!this.device) {
      throw Error("Failed to request WebGPU device.");
    }

    this.device.lost.then((info) => {
      if (info.reason !== "destroyed") {
        throw new Error(`WebGPU device was lost: ${info.message}`);
      }
    });
  }

  private getCanvas() {
    this.canvas = document.querySelector("canvas");
    if (!this.canvas) {
      throw Error("Failed to find canvas element.");
    }

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const canvas = entry.target as HTMLCanvasElement;
        const width = entry.contentBoxSize[0].inlineSize;
        const height = entry.contentBoxSize[0].blockSize;
        canvas.width = Math.max(
          1,
          Math.min(width, this.device.limits.maxTextureDimension2D)
        );
        canvas.height = Math.max(
          1,
          Math.min(height, this.device.limits.maxTextureDimension2D)
        );
      }
    });
    observer.observe(this.canvas);
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

  private createRenderPipeline() {
    const shaderModule = this.createShaderModule(
      "Triangle Shader Module",
      shaderCode
    );

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
      label: "Triangle Render Pipeline",
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

  private createVertexBuffer() {
    this.vertexBuffer = this.createBuffer(
      "Vertex Buffer",
      vertices.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(
      this.vertexBuffer,
      0,
      vertices,
      0,
      vertices.length
    );
  }

  private createUniformBuffer() {
    const uniformBufferSize = (2 + 2) * Float32Array.BYTES_PER_ELEMENT;
    this.uniformValues = new Float32Array(
      uniformBufferSize / Float32Array.BYTES_PER_ELEMENT
    );
    this.scaleOffset = 0;
    this.offsetOffset = 2;
    this.uniformValues.set([-0.5, -0.25], this.offsetOffset);

    this.uniformBuffer = this.createBuffer(
      "Triangle Uniform Buffer",
      uniformBufferSize,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    this.bindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer,
          },
        },
      ],
    });
  }

  private createBuffer(
    label: string,
    size: number,
    usage: GPUBufferUsageFlags
  ) {
    const buffer = this.device.createBuffer({
      label,
      size,
      usage,
    });

    return buffer;
  }

  private createShaderModule(label: string, code: string) {
    const shaderModule = this.device.createShaderModule({
      label,
      code,
    });

    return shaderModule;
  }

  private run() {
    requestAnimationFrame(this.drawFrame);
  }

  private drawFrame() {
    const aspect = this.canvas.width / this.canvas.height;
    this.uniformValues.set([0.5 / aspect, 0.5], this.scaleOffset);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformValues);

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context
      .getCurrentTexture()
      .createView({ label: "Render Target Texture View" });
    const colorAttachments: GPURenderPassColorAttachment[] = [
      {
        view: textureView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ];

    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "Triangle Renderpass",
      colorAttachments,
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipeline);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.draw(3);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(this.drawFrame);
  }
}
