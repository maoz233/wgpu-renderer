import path from 'path';
import webpack from 'webpack';

const HtmlWebpackPlugin = require('html-webpack-plugin');

const config: webpack.Configuration = {
  mode: "development",
  entry: "./src/index.ts",
  devtool: 'inline-source-map',
  plugins: [
    new HtmlWebpackPlugin({
      title: "Custom Template",
      template: "./src/index.html",
    })
  ],
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ]
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "index.bundle.js",
    clean: true,
  },
};

export default config;
