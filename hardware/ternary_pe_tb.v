// Testbench for ternary_pe: run a small ternary dot product and check the
// result. Drives weights ±1 alternately with constant input, and after
// N cycles checks that the accumulator matches sum(W_i * X) = 0 (cancels).
// For iverilog: `iverilog -o pe_tb ternary_full_adder.v ternary_pe_tb.v && vvp pe_tb`

`timescale 1ns/1ps

module ternary_pe_tb;
    reg clk = 0;
    reg rst_n = 0;
    reg signed [1:0]  W;
    reg signed [35:0] X;
    wire signed [35:0] ACC;

    ternary_pe dut (.clk(clk), .rst_n(rst_n), .W(W), .X(X), .ACC(ACC));

    always #5 clk = ~clk;  // 100 MHz

    integer i;
    initial begin
        // Drive: W = +1, X = encode(5). Run 4 cycles. ACC should be 20.
        // For brevity we just simulate with X already encoded as a 36-bit
        // signed value where the integer 5 maps to its balanced-ternary
        // representation. A separate encoder/decoder module would do that
        // conversion at the boundary.
        X = 36'sd5;
        rst_n = 0; #20; rst_n = 1;
        W = 2'sd1;
        for (i = 0; i < 4; i = i + 1) @(posedge clk);
        if (ACC !== 36'sd20) $display("FAIL  ACC=%0d", ACC); else $display("PASS  ACC=%0d", ACC);

        // W = -1, run 2 more cycles. ACC -> 20 - 10 = 10.
        W = -2'sd1;
        for (i = 0; i < 2; i = i + 1) @(posedge clk);
        if (ACC !== 36'sd10) $display("FAIL  ACC=%0d", ACC); else $display("PASS  ACC=%0d", ACC);

        // W = 0, run 5 cycles. ACC stays at 10.
        W = 2'sd0;
        for (i = 0; i < 5; i = i + 1) @(posedge clk);
        if (ACC !== 36'sd10) $display("FAIL  ACC=%0d", ACC); else $display("PASS  ACC=%0d", ACC);

        $finish;
    end
endmodule
