import Renderer from "@/renderer";
const renderer = new Renderer();

try {
  renderer.render()
} catch (err: any) {
  if (err instanceof Error) {
    console.error(err.message)
  }
}
