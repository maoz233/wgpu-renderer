import shaderCode from "@/shaders/shader.wgsl";
import { mat4 } from "wgpu-matrix";
import { GUI, GUIController } from "dat.gui";
import { rand } from "@/utils";

type Transform = {
  offset: number[];
  scale: number;
};

export default class Renderer {
  private adapter: GPUAdapter;
  private device: GPUDevice;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;
  private renderPipeline: GPURenderPipeline;
  private vertexBuffer: GPUBuffer;
  private transforms: Array<Transform>;
  private storageValues: Float32Array;
  private storageBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup;
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
    this.createStorageBuffer();
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
    const shaderModule = this.createShaderModule("Shader Module", shaderCode);

    this.renderPipeline = this.device.createRenderPipeline({
      label: "Render Pipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride:
              4 *
              (Float32Array.BYTES_PER_ELEMENT + Uint8Array.BYTES_PER_ELEMENT),
            stepMode: "vertex" as GPUVertexStepMode,
            attributes: [
              {
                format: "float32x4" as GPUVertexFormat,
                offset: 0,
                shaderLocation: 0,
              },
              {
                format: "unorm8x4" as GPUVertexFormat,
                offset: 4 * Float32Array.BYTES_PER_ELEMENT,
                shaderLocation: 1,
              },
            ],
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: this.presentationFormat,
          },
        ],
      },
    });
  }

  private createVertexBuffer() {
    const vertexCount = 3;
    const vertexComponents = 4;
    const colorComponents = 4;
    const unitSize =
      vertexComponents * Float32Array.BYTES_PER_ELEMENT +
      colorComponents * Uint8Array.BYTES_PER_ELEMENT;

    const vertexArrayBuffer = new ArrayBuffer(vertexCount * unitSize);
    let vertexOffset = 0;
    let vertexData = new Float32Array(vertexArrayBuffer);
    let colorOffset = vertexComponents * Float32Array.BYTES_PER_ELEMENT;
    let colorData = new Uint8Array(vertexArrayBuffer);

    const data = [
      {
        vertices: [0.0, 0.5, 0.0, 1.0],
        color: [255, 0, 0, 255],
      },
      {
        vertices: [-0.5, -0.5, 0.0, 1.0],
        color: [0, 255, 0, 255],
      },
      {
        vertices: [0.5, -0.5, 0.0, 1.0],
        color: [0, 0, 255, 255],
      },
    ];

    for (let i = 0; i < vertexCount; ++i) {
      vertexData.set(data[i].vertices, vertexOffset);
      colorData.set(data[i].color, colorOffset);

      colorOffset += unitSize / Uint8Array.BYTES_PER_ELEMENT;
      vertexOffset += unitSize / Float32Array.BYTES_PER_ELEMENT;
    }

    this.vertexBuffer = this.createBuffer(
      "Vertex Buffer",
      vertexArrayBuffer.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    );

    this.device.queue.writeBuffer(
      this.vertexBuffer,
      0,
      vertexArrayBuffer,
      0,
      vertexArrayBuffer.byteLength
    );
  }

  private createStorageBuffer() {
    const storageBufferSize = 4 * 4 * Float32Array.BYTES_PER_ELEMENT * 100;

    this.storageValues = new Float32Array(
      storageBufferSize / Float32Array.BYTES_PER_ELEMENT
    );

    this.storageBuffer = this.createBuffer(
      `Transform Storage Buffer`,
      storageBufferSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    this.bindGroup = this.device.createBindGroup({
      label: `Transform Bind Group`,
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.storageBuffer,
          },
        },
      ],
    });

    this.transforms = new Array<Transform>();
    for (let i = 0; i < 100; ++i) {
      this.transforms.push({
        offset: [rand(-0.9, 0.9), rand(-0.9, 0.9)],
        scale: rand(0.2, 0.5),
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
      label: "Renderpass Descriptor",
      colorAttachments,
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipeline);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);

    this.transforms.forEach(({ offset, scale }: Transform, index: number) => {
      const arrayOffset = index * (4 * 4);

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
      this.storageValues.set(transform, arrayOffset);
    });
    this.device.queue.writeBuffer(this.storageBuffer, 0, this.storageValues);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.draw(3, 100);

    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(this.drawFrame);
  }
}
