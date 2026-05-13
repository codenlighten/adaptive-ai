// 1-D systolic ternary matmul array.
//
// N processing elements arranged in a chain. At each clock cycle:
//   - Input x flows from west to east (PE_0 -> PE_1 -> ... -> PE_{N-1}).
//   - Each PE multiplies x by its current ternary weight (no actual
//     multiplier — sign-flip and add) and accumulates into its local
//     accumulator.
//
// To compute y[i] = sum_j W[i][j] * x[j] for one row of an N-column
// matrix, you load W[i][0..N-1] into the PEs' weight registers, stream
// x[0..N-1] through the array, and after N cycles each PE holds its
// partial contribution. The y[i] is read by summing the PE accumulators
// (a tree-adder, omitted here for brevity).
//
// This is the same architecture used in Google's TPU and academic
// neural accelerators — instantiated here with ternary PEs instead of
// fp16/bf16 multipliers. Synthesis target: iCE40 or ECP5.

`timescale 1ns/1ps

module ternary_systolic_pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        ld_w,         // load weight enable
    input  wire signed [1:0]  w_in,
    input  wire signed [35:0] x_in,
    output reg  signed [35:0] x_out,  // forwarded x for next PE
    output wire signed [35:0] acc
);
    reg signed [1:0]  w_reg;
    reg signed [35:0] acc_reg;

    wire signed [35:0] addend = (w_reg == 2'sd1)  ? x_in
                              : (w_reg == -2'sd1) ? -x_in
                              :                     36'sd0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_reg   <= 2'sd0;
            acc_reg <= 36'sd0;
            x_out   <= 36'sd0;
        end else begin
            if (ld_w) w_reg <= w_in;
            acc_reg <= acc_reg + addend;
            x_out   <= x_in;            // pipeline: forward to next PE
        end
    end

    assign acc = acc_reg;
endmodule


// Parameterizable N-element 1-D array. Each PE handles one column of W.
// Ports are packed bit-vectors so this is Verilog-2001-clean.
module ternary_systolic_array #(
    parameter integer N = 8
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  ld_w,
    input  wire [2*N-1:0]        w_in_flat,    // N weights packed (2 bits each)
    input  wire signed [35:0]    x_in,         // streamed in from the west
    output wire [36*N-1:0]       acc_flat      // per-PE accumulators packed
);
    wire signed [35:0] x_pipe [0:N];
    assign x_pipe[0] = x_in;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : pe
            wire signed [35:0] acc_i;
            ternary_systolic_pe u (
                .clk   (clk),
                .rst_n (rst_n),
                .ld_w  (ld_w),
                .w_in  (w_in_flat[2*i +: 2]),
                .x_in  (x_pipe[i]),
                .x_out (x_pipe[i+1]),
                .acc   (acc_i)
            );
            assign acc_flat[36*i +: 36] = acc_i;
        end
    endgenerate

endmodule
