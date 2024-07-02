import shaderCode from "@/shaders/shader.wgsl";
import { mat4 } from "wgpu-matrix";
import { GUI, GUIController } from "dat.gui";
import { rand } from "@/utils";

const vertices = new Float32Array([
  0.0, 0.6, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -0.5, -0.6, 0.0, 1.0, 0.0, 1.0, 0.0,
  1.0, 0.5, -0.6, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
]);

type Transform = {
  buffer: GPUBuffer;
  offset: number[];
  scale: number;
  bindGroup: GPUBindGroup;
};

export default class Renderer {
  private adapter: GPUAdapter;
  private device: GPUDevice;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;
  private vertexBuffer: GPUBuffer;
  private uniforms: Transform[];
  private renderPipeline: GPURenderPipeline;
  private current: number;
  private profilerController: GUIController;

  public constructor() {
    this.current = Date.now();
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
    this.initGUI();
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
    this.uniforms = [];

    const transformBufferSize = 4 * 4 * Float32Array.BYTES_PER_ELEMENT;

    for (let i = 0; i < 100; ++i) {
      const transformBuffer = this.createBuffer(
        `Transform Uniform Buffer ${i}`,
        transformBufferSize,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );

      const bindGroup = this.device.createBindGroup({
        label: `Transform Uniform Bind Group ${i}`,
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: transformBuffer,
            },
          },
        ],
      });

      this.uniforms.push({
        buffer: transformBuffer,
        offset: [rand(-0.9, 0.9), rand(-0.9, 0.9)],
        scale: rand(0.2, 0.5),
        bindGroup,
      });
    }
  }

  private createShaderModule(label: string, code: string) {
    const shaderModule = this.device.createShaderModule({
      label,
      code,
    });

    return shaderModule;
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

  private initGUI() {
    const profiler = { fps: "0" };
    const gui = new GUI({
      name: "My GUI",
      autoPlace: true,
      hideable: true,
    });
    this.profilerController = gui.add(profiler, "fps").name("FPS");
  }

  private run() {
    requestAnimationFrame(this.drawFrame);
  }

  private drawFrame() {
    const now = Date.now();
    this.profilerController.setValue((1000 / (now - this.current)).toFixed(2));
    this.current = now;

    // model-view-projectin matrix
    const aspect = this.canvas.width / this.canvas.height;
    const view = mat4.lookAt(
      [0.0, 0.0, -3.0],
      [0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0]
    );
    const projection = mat4.perspective((2 * Math.PI) / 12, aspect, 1.0, 100.0);

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

    for (const { buffer, offset, scale, bindGroup } of this.uniforms) {
      const scaleMat = mat4.scale(mat4.identity(), [
        scale / aspect,
        scale,
        1.0,
        1.0,
      ]);
      const translateMat = mat4.translate(mat4.identity(), [
        ...offset,
        1.0,
        1.0,
      ]);
      const rotateMat = mat4.rotateY(
        mat4.identity(),
        (this.current / 1000) % 360
      );
      const model = mat4.mul(scaleMat, mat4.mul(translateMat, rotateMat));
      const transform = mat4.multiply(projection, mat4.multiply(view, model));
      this.device.queue.writeBuffer(
        buffer,
        0,
        transform.buffer,
        transform.byteOffset,
        transform.byteLength
      );

      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.draw(3);
    }

    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(this.drawFrame);
  }
}
