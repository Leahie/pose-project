export const css = `
.skeletonCanvas_root {
  width: 100vw;
  height: 100vh;
  background: #050810;
  position: relative;
}

.skeletonCanvas_overlay {
  position: absolute;
  top: 16px;
  right: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: flex-end;
  color: white;
  font-family: monospace;
  font-size: 12px;
}

.skeletonCanvas_overlayText {
  opacity: 0.6;
}

.skeletonCanvas_button {
  padding: 6px 14px;
  background: rgba(255,255,255,0.08);
  color: #aaa;
  border: 1px solid #333;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}
`

export const classNames = {
  root: 'skeletonCanvas_root',
  overlay: 'skeletonCanvas_overlay',
  overlayText: 'skeletonCanvas_overlayText',
  button: 'skeletonCanvas_button',
}
