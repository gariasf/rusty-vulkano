#version 460

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Descriptor. Buffer called buf
layout (set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint index = gl_GlobalInvocationId.x;
    buf.data[index] *= 12;
}