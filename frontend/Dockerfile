FROM oven/bun:1

# Set working directory
WORKDIR /app

# Copy Bun files first
COPY package.json tsconfig.json ./

# Install dependencies
RUN bun install

# Copy the rest of the app
COPY . .

# Expose the default port Bun listens on
EXPOSE 5173

CMD ["bun", "run", "dev"]