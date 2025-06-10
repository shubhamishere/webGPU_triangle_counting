// This WGSL file contains the parallel algorithm for triangle counting.
// It's designed to run on the GPU, with one thread handling one node of the graph.

// The data buffers are bound to the shader. These must match the bind group layout in main.js.
// binding 0: row_ptr - The Compressed Sparse Row (CSR) offset array.
//            row_ptr[i] gives the starting index in the edge_list for node i's neighbors.
// binding 1: edge_list - The CSR column index array. A flattened list of all graph edges.
// binding 2: triangle_count - An atomic counter where each thread will add the triangles it finds.
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> edge_list: array<u32>;
@group(0) @binding(2) var<storage, read_write> total_triangles: atomic<u32>;

// This function performs a binary search to find if 'searchValue' exists in the neighbor list of 'node'.
// equivalent of the is_neighbor() check.
// it's a fundamental operation for the "node-iterator" approach.
fn is_neighbor(node: u32, searchValue: u32) -> bool {
    var start = row_ptr[node];
    var end = row_ptr[node + 1u];

    // Standard binary search on a sorted list.
    while (start < end) {
        let mid = start + (end - start) / 2u;
        let val = edge_list[mid];
        if (val == searchValue) {
            return true;
        } else if (val < searchValue) {
            start = mid + 1u;
        } else {
            end = mid;
        }
    }
    return false;
}

// number of threads in a workgroup
@workgroup_size(256)
@compute
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread is responsible for one node, 'u'.
    let u = global_id.x;

    // Get the number of nodes in the graph. arrayLength gives the total size of the buffer.
    // The number of nodes is one less than the length of the row_ptr array.
    let num_nodes = arrayLength(&row_ptr) - 1u;

    // Boundary check: ensure the thread is not processing a non-existent node.
    if (u >= num_nodes) {
        return;
    }

    // Get the slice of the edge_list that contains the neighbors of node 'u'.
    let start_u = row_ptr[u];
    let end_u = row_ptr[u + 1u];

    // Iterate through all neighbors 'v' of 'u'.
    for (var i = start_u; i < end_u; i = i + 1u) {
        let v = edge_list[i];

        // To avoid duplicate counting and self-loops, we enforce a strict ordering: u < v < w.
        // We only proceed if the neighbor 'v' has a higher index than 'u'.
        if (v <= u) {
            continue;
        }

        // Get the slice for neighbors of 'v'.
        let start_v = row_ptr[v];
        let end_v = row_ptr[v + 1u];

        // This is the core logic: Find common neighbors between u and v.
        // We iterate through neighbors 'w' of 'u' and 'v' simultaneously.
        // This is a "merge join" or "sorted list intersection" algorithm.
        var ptr_u = i + 1u; // Start checking u's neighbors from the one after v
        var ptr_v = start_v;

        while (ptr_u < end_u && ptr_v < end_v) {
            let w_from_u = edge_list[ptr_u];
            let w_from_v = edge_list[ptr_v];

            if (w_from_u == w_from_v) {
                // Common neighbor found! This forms a triangle (u, v, w).
                // Because we enforced u < v and w > v (from w_from_u > v), the triangle is unique.
                atomicAdd(&total_triangles, 1u);
                ptr_u = ptr_u + 1u;
                ptr_v = ptr_v + 1u;
            } else if (w_from_u < w_from_v) {
                // Advance the pointer for u's neighbors.
                ptr_u = ptr_u + 1u;
            } else {
                // Advance the pointer for v's neighbors.
                ptr_v = ptr_v + 1u;
            }
        }
    }
}
