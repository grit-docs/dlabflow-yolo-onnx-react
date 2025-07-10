import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  plugins: [
    react(),
    basicSsl(), // 안정적인 SSL 플러그인
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.{wasm,mjs}',
          dest: 'ort-wasm-files'
        }
      ]
    })
  ],
  server: {
    host: '0.0.0.0',
    port: 5173,
    https: {
      // 더 호환성 좋은 SSL 설정
      minVersion: 'TLSv1.2',
      ciphers: 'HIGH:!aNULL:!MD5',
    },
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  assetsInclude: ['**/*.wasm'],
})
