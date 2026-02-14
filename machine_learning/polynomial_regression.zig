const std = @import("std");
const Allocator = std.mem.Allocator;

pub const PolynomialRegressionError = error{
    NegativeDegree,
    NotFullRank,
    NotFitted,
    InvalidDimensions,
    OutOfMemory,
    SingularMatrix,
};

pub fn PolynomialRegression(comptime T: type) type {
    return struct {
        const Self = @This();

        degree: usize,
        params: ?[]T,
        allocator: Allocator,

        pub fn init(allocator: Allocator, degree: usize) Self {
            return .{
                .degree = degree,
                .params = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.params) |params| {
                self.allocator.free(params);
                self.params = null;
            }
        }

        fn createDesignMatrix(self: *Self, data: []const T) ![][]T {
            const n = data.len;
            const m = self.degree + 1;

            const matrix = try self.allocator.alloc([]T, n);
            errdefer self.allocator.free(matrix);

            for (0..n) |i| {
                matrix[i] = try self.allocator.alloc(T, m);
                errdefer {
                    for (0..i + 1) |j| {
                        self.allocator.free(matrix[j]);
                    }
                }
            }

            for (data, 0..) |x, i| {
                var power: T = 1.0;
                for (0..m) |j| {
                    matrix[i][j] = power;
                    power *= x;
                }
            }

            return matrix;
        }

        fn freeMatrix(self: *Self, matrix: [][]T) void {
            for (matrix) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(matrix);
        }

        fn transposeMultiply(self: *Self, X: []const []const T) ![][]T {
            const n = X.len;
            const m = X[0].len;

            const result = try self.allocator.alloc([]T, m);
            errdefer self.allocator.free(result);

            for (0..m) |i| {
                result[i] = try self.allocator.alloc(T, m);
                errdefer {
                    for (0..i + 1) |j| {
                        self.allocator.free(result[j]);
                    }
                }
                for (0..m) |j| {
                    var sum: T = 0.0;
                    for (0..n) |k| {
                        sum += X[k][i] * X[k][j];
                    }
                    result[i][j] = sum;
                }
            }

            return result;
        }

        fn invertMatrix(self: *Self, A: []const []const T) ![][]T {
            const n = A.len;
            const inv = try self.allocator.alloc([]T, n);
            errdefer self.allocator.free(inv);

            for (0..n) |i| {
                inv[i] = try self.allocator.alloc(T, n);
                errdefer {
                    for (0..i + 1) |j| {
                        self.allocator.free(inv[j]);
                    }
                }
                for (0..n) |j| {
                    inv[i][j] = if (i == j) 1.0 else 0.0;
                }
            }

            const temp = try self.allocator.alloc([]T, n);
            defer self.allocator.free(temp);
            for (0..n) |i| {
                temp[i] = try self.allocator.alloc(T, n);
                for (0..n) |j| {
                    temp[i][j] = A[i][j];
                }
            }
            defer {
                for (temp) |row| {
                    self.allocator.free(row);
                }
            }

            for (0..n) |i| {
                var max_row = i;
                for (i + 1..n) |k| {
                    if (@abs(temp[k][i]) > @abs(temp[max_row][i])) {
                        max_row = k;
                    }
                }

                if (@abs(temp[max_row][i]) < 1e-10) {
                    return PolynomialRegressionError.SingularMatrix;
                }

                if (max_row != i) {
                    const tmp_row = temp[i];
                    temp[i] = temp[max_row];
                    temp[max_row] = tmp_row;

                    const tmp_inv = inv[i];
                    inv[i] = inv[max_row];
                    inv[max_row] = tmp_inv;
                }

                const pivot = temp[i][i];
                for (0..n) |j| {
                    temp[i][j] /= pivot;
                    inv[i][j] /= pivot;
                }

                for (0..n) |k| {
                    if (k != i) {
                        const factor = temp[k][i];
                        for (0..n) |j| {
                            temp[k][j] -= factor * temp[i][j];
                            inv[k][j] -= factor * inv[i][j];
                        }
                    }
                }
            }

            return inv;
        }

        pub fn fit(self: *Self, x_train: []const T, y_train: []const T) !void {
            if (x_train.len != y_train.len) {
                return PolynomialRegressionError.InvalidDimensions;
            }

            if (x_train.len <= self.degree) {
                return PolynomialRegressionError.NotFullRank;
            }

            const X = try self.createDesignMatrix(x_train);
            defer self.freeMatrix(X);

            const XtX = try self.transposeMultiply(X);
            defer self.freeMatrix(XtX);

            const XtX_inv = try self.invertMatrix(XtX);
            defer self.freeMatrix(XtX_inv);

            const m = self.degree + 1;
            const Xty = try self.allocator.alloc(T, m);
            defer self.allocator.free(Xty);

            for (0..m) |i| {
                var sum: T = 0.0;
                for (0..x_train.len) |j| {
                    sum += X[j][i] * y_train[j];
                }
                Xty[i] = sum;
            }

            if (self.params) |params| {
                self.allocator.free(params);
            }

            self.params = try self.allocator.alloc(T, m);
            for (0..m) |i| {
                var sum: T = 0.0;
                for (0..m) |j| {
                    sum += XtX_inv[i][j] * Xty[j];
                }
                self.params.?[i] = sum;
            }
        }

        pub fn predict(self: *Self, x: []const T) ![]T {
            if (self.params == null) {
                return PolynomialRegressionError.NotFitted;
            }

            const X = try self.createDesignMatrix(x);
            defer self.freeMatrix(X);

            const result = try self.allocator.alloc(T, x.len);
            errdefer self.allocator.free(result);

            for (0..x.len) |i| {
                var sum: T = 0.0;
                for (0..self.params.?.len) |j| {
                    sum += X[i][j] * self.params.?[j];
                }
                result[i] = sum;
            }

            return result;
        }
    };
}

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

test "polynomial regression - degree 0 (constant)" {
    const allocator = testing.allocator;
    var poly_reg = PolynomialRegression(f64).init(allocator, 0);
    defer poly_reg.deinit();

    const x_train = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_train = [_]f64{ 5.0, 5.0, 5.0, 5.0, 5.0 };

    try poly_reg.fit(&x_train, &y_train);

    const x_test = [_]f64{6.0};
    const predictions = try poly_reg.predict(&x_test);
    defer allocator.free(predictions);

    try expectApproxEqAbs(5.0, predictions[0], 1e-6);
}

test "polynomial regression - degree 1 (linear)" {
    const allocator = testing.allocator;
    var poly_reg = PolynomialRegression(f64).init(allocator, 1);
    defer poly_reg.deinit();

    const x_train = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_train = [_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 };

    try poly_reg.fit(&x_train, &y_train);

    try expectApproxEqAbs(1.0, poly_reg.params.?[0], 1e-6);
    try expectApproxEqAbs(2.0, poly_reg.params.?[1], 1e-6);

    const x_test = [_]f64{5.0};
    const predictions = try poly_reg.predict(&x_test);
    defer allocator.free(predictions);

    try expectApproxEqAbs(11.0, predictions[0], 1e-6);
}

test "polynomial regression - degree 2 (quadratic)" {
    const allocator = testing.allocator;
    var poly_reg = PolynomialRegression(f64).init(allocator, 2);
    defer poly_reg.deinit();

    const x_train = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_train = [_]f64{ 3.0, 6.0, 13.0, 24.0, 39.0 };

    try poly_reg.fit(&x_train, &y_train);

    try expectApproxEqAbs(3.0, poly_reg.params.?[0], 1e-6);
    try expectApproxEqAbs(1.0, poly_reg.params.?[1], 1e-6);
    try expectApproxEqAbs(2.0, poly_reg.params.?[2], 1e-6);

    const x_test = [_]f64{5.0};
    const predictions = try poly_reg.predict(&x_test);
    defer allocator.free(predictions);

    try expectApproxEqAbs(58.0, predictions[0], 1e-6);
}

test "polynomial regression - degree 3 (cubic)" {
    const allocator = testing.allocator;
    var poly_reg = PolynomialRegression(f64).init(allocator, 3);
    defer poly_reg.deinit();

    const x_train = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_train = [_]f64{ -5.0, -3.0, 1.0, 13.0, 39.0, 85.0 };

    try poly_reg.fit(&x_train, &y_train);

    try expectApproxEqAbs(-5.0, poly_reg.params.?[0], 1e-6);
    try expectApproxEqAbs(3.0, poly_reg.params.?[1], 1e-6);
    try expectApproxEqAbs(-2.0, poly_reg.params.?[2], 1e-6);
    try expectApproxEqAbs(1.0, poly_reg.params.?[3], 1e-6);

    const x_test = [_]f64{6.0};
    const predictions = try poly_reg.predict(&x_test);
    defer allocator.free(predictions);

    try expectApproxEqAbs(157.0, predictions[0], 1e-6);
}

test "polynomial regression - multiple predictions" {
    const allocator = testing.allocator;
    var poly_reg = PolynomialRegression(f64).init(allocator, 2);
    defer poly_reg.deinit();

    const x_train = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const y_train = [_]f64{ 0.0, 1.0, 4.0, 9.0 };

    try poly_reg.fit(&x_train, &y_train);

    const x_test = [_]f64{ 4.0, 5.0, 6.0 };
    const predictions = try poly_reg.predict(&x_test);
    defer allocator.free(predictions);

    try expectApproxEqAbs(16.0, predictions[0], 1e-6);
    try expectApproxEqAbs(25.0, predictions[1], 1e-6);
    try expectApproxEqAbs(36.0, predictions[2], 1e-6);
}

test "polynomial regression - error on predict before fit" {
    const allocator = testing.allocator;
    var poly_reg = PolynomialRegression(f64).init(allocator, 2);
    defer poly_reg.deinit();

    const x_test = [_]f64{1.0};
    const result = poly_reg.predict(&x_test);

    try expectEqual(PolynomialRegressionError.NotFitted, result);
}

test "polynomial regression - error on insufficient data" {
    const allocator = testing.allocator;
    var poly_reg = PolynomialRegression(f64).init(allocator, 3);
    defer poly_reg.deinit();

    const x_train = [_]f64{ 0.0, 1.0 };
    const y_train = [_]f64{ 0.0, 1.0 };

    const result = poly_reg.fit(&x_train, &y_train);

    try expectEqual(PolynomialRegressionError.NotFullRank, result);
}
