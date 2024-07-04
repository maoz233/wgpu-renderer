export function rand(min?: number, max?: number) {
  if (undefined === min) {
    min = 0;
    max = 1;
  } else if (undefined === max) {
    max = min;
    min = 0;
  }

  return min + Math.random() * (max - min);
}

export async function loadImageBitmap(url: string) {
  const res = await fetch(url);
  const blob = await res.blob();

  return await createImageBitmap(blob, { colorSpaceConversion: "none" });
}
