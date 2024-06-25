const vertices = new Float32Array([0.0, 0.6, 0, 1, 1, 0, 0, 1, -0.5, -0.6, 0, 1, 0, 1, 0, 1, 0.5, -0.6, 0, 1, 0, 0, 1, 1,]);

const shaderCode = `
struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f
}

@vertex
fn vertex_main(@location(0) position: vec4f,
              @location(1) color: vec4f) -> VertexOut
{
  var output : VertexOut;
  output.position = position;
  output.color = color;
  return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
  return fragData.color;
}
`;

export default class Renderer {
  adapter: GPUAdapter;
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  presentationFormat: GPUTextureFormat;
  vertexBuffer: GPUBuffer;
  renderPipeline: GPURenderPipeline;

  constructor() { }

  async render() {
    await this.init();
    this.run();
  }

  async init() {
    this.checkWebGPUSupport();
    await this.requestAdapter();
    await this.requestDevice();
    this.getCanvas();
    this.getContext();
    this.configContext(this.device, this.presentationFormat, "premultiplied");
    this.vertexBuffer = this.createBuffer("Vertex Buffer", vertices, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    this.createRenderPipeline();
  }

  checkWebGPUSupport() {
    if (!navigator.gpu) {
      throw Error("WebGPU not supported.");
    }
  }

  async requestAdapter() {
    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw Error("Failed to request WebGPU adapter.");
    }
  }

  async requestDevice() {
    this.device = await this.adapter.requestDevice();
    if (!this.device) {
      throw Error("Failed to request WebGPU device.");
    }
  }

  getCanvas() {
    this.canvas = document.querySelector('canvas');
    if (!this.canvas) {
      throw Error("Failed to find canvas element.");
    }
  }

  getContext() {
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) {
      throw Error("Failed to get WebGPU context from canvas.");
    }

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  }

  configContext(device: GPUDevice, format: GPUTextureFormat, alphaMode: GPUCanvasAlphaMode) {
    this.context.configure({
      device,
      format,
      alphaMode,
    });
  }

  createBuffer(label: string, data: Float32Array, usage: GPUBufferUsageFlags) {
    const buffer = this.device.createBuffer({
      label,
      size: data.byteLength,
      usage,
    });
    this.device.queue.writeBuffer(buffer, 0, data, 0, data.length);

    return buffer;
  }

  createRenderPipeline() {
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

  createShaderModule(label: string, code: string) {
    const shaderModule = this.device.createShaderModule({
      label,
      code,
    })

    return shaderModule;
  }

  run() {
    requestAnimationFrame(() => this.drawFrame());
  }

  drawFrame() {
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
