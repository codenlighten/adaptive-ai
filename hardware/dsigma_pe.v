// Delta-Sigma Processing Element
//
// Wraps the existing ternary_pe with a per-weight shift register that
// streams a delta-sigma encoded trit per cycle. Each weight position
// stores T pre-computed trits in a circular buffer; at each cycle the
// next trit drives the existing ternary signed-adder.
//
// After T cycles (and one division by T), the accumulator holds
// alpha * (W_target @ x) where W_target is the original continuous
// weight that the modulator was encoding.
//
// Storage per weight: T trits (we store them packed; for T=8 that's
// 16 bits = 2 bytes per weight). Compute per cycle: one signed add,
// zero multiplications.
//
// Anytime inference: stop after k < T cycles for a noisier estimate.
// The early-stopping trit pattern is exactly the first k bits of the
// stream — no recomputation needed.
//
// This file imports the ternary_full_adder from ternary_full_adder.v.

`timescale 1ns/1ps

module dsigma_pe #(
    parameter integer T_MAX = 8           // max delta-sigma stream length
) (
    input  wire                   clk,
    input  wire                   rst_n,
    // ld_stream: load the next trit into the shift register on this cycle
    input  wire                   ld_stream,
    input  wire signed [1:0]      w_in_trit,    // one trit of the stream
    // run: shift one trit out of the stream and accumulate
    input  wire                   run,
    // clear: zero the accumulator (between matmul rows)
    input  wire                   clear,
    input  wire signed [35:0]     x_in,
    output reg  signed [35:0]     acc           // accumulated y for this row
);

    // Circular trit storage (T_MAX trits, 2 bits each)
    reg signed [1:0] stream [0:T_MAX-1];
    reg [$clog2(T_MAX)-1:0] head;     // write pointer for loading
    reg [$clog2(T_MAX)-1:0] tail;     // read pointer for running

    integer i;
    initial begin
        for (i = 0; i < T_MAX; i = i + 1) stream[i] = 2'sd0;
    end

    // Compute addend from current head trit
    wire signed [1:0] current_w = stream[tail];
    wire signed [35:0] addend = (current_w == 2'sd1)  ? x_in
                              : (current_w == -2'sd1) ? -x_in
                              :                          36'sd0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc  <= 36'sd0;
            head <= 0;
            tail <= 0;
        end else if (clear) begin
            acc  <= 36'sd0;
            tail <= 0;
        end else begin
            if (ld_stream) begin
                stream[head] <= w_in_trit;
                head <= head + 1;
            end
            if (run) begin
                acc  <= acc + addend;
                tail <= tail + 1;
            end
        end
    end

endmodule
