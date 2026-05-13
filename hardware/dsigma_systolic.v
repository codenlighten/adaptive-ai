// 1-D systolic array of N delta-sigma processing elements.
//
// Each PE holds its own length-T_MAX delta-sigma stream. The input x
// flows west-to-east through the chain; at each cycle every PE
// accumulates one trit's contribution to its row's dot product.
//
// To compute y = W @ x for an N-column W:
//   1. Pre-encode each row of W as a T-step stream and load into the
//      corresponding PE's buffer.
//   2. Stream x[j] for j = 0..N-1 through the chain.
//   3. After T*N cycles, divide each PE's accumulator by T to get y[i].
//
// Anytime inference: at any k < T cycles, the partial-sum gives an
// O(1/k)-accurate estimate.

`timescale 1ns/1ps

module dsigma_systolic #(
    parameter integer N      = 8,    // number of PEs (one per column of W)
    parameter integer T_MAX  = 8     // max delta-sigma stream length per PE
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   clear,
    input  wire                   ld_stream,
    input  wire signed [1:0]      w_in_trit,
    input  wire [3:0]             ld_pe_select,    // which PE to load (for small N)
    input  wire                   run,
    input  wire signed [35:0]     x_in,
    output wire [36*N-1:0]        acc_flat
);

    wire signed [35:0] x_pipe [0:N];
    assign x_pipe[0] = x_in;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : pe
            wire signed [35:0] acc_i;
            wire ld_this = ld_stream && (ld_pe_select == i[3:0]);
            dsigma_pe #(.T_MAX(T_MAX)) u (
                .clk         (clk),
                .rst_n       (rst_n),
                .ld_stream   (ld_this),
                .w_in_trit   (w_in_trit),
                .run         (run),
                .clear       (clear),
                .x_in        (x_pipe[i]),
                .acc         (acc_i)
            );
            assign acc_flat[36*i +: 36] = acc_i;
            // Pipeline x to next PE — uses one D-FF per bit
            reg signed [35:0] x_next;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) x_next <= 36'sd0;
                else if (run) x_next <= x_pipe[i];
            end
            assign x_pipe[i+1] = x_next;
        end
    endgenerate

endmodule
