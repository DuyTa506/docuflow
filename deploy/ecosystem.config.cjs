/**
 * PM2 config for DocuFlow Angular production build (static SPA).
 *
 * Start:  pm2 start deploy/ecosystem.config.cjs
 * Or use: bash deploy/install-autostart.sh
 */
const path = require("path");

const root = path.resolve(__dirname, "..");
const dist = path.join(root, "Fe-Library", "dist");

module.exports = {
  apps: [
    {
      name: "docuflow-fe",
      script: "serve",
      cwd: root,
      env: {
        PM2_SERVE_PATH: dist,
        PM2_SERVE_PORT: 4200,
        PM2_SERVE_SPA: "true",
        PM2_SERVE_HOMEPAGE: "/index.html",
      },
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
    },
  ],
};
