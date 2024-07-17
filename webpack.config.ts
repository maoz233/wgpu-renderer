import path from "path";
import webpack from "webpack";
import "webpack-dev-server";

const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");

const config: webpack.Configuration = {
  mode: "development",
  entry: path.resolve(__dirname, "src/index.ts"),
  devtool: "inline-source-map",
  plugins: [
    new HtmlWebpackPlugin({
      title: "Custom Template",
      template: path.resolve(__dirname, "src/index.html"),
    }),
    new CopyWebpackPlugin({
      patterns: [
        { from: "src/style.css", to: "style.css" },
        { from: "public", to: "" },
      ],
    }),
  ],
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.wgsl$/,
        use: "raw-loader",
      },
    ],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
    extensions: [".ts", ".js"],
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "index.bundle.js",
    clean: true,
  },
  devServer: {
    // open: true,
  },
};

export default config;
