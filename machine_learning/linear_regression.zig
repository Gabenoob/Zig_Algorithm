const std = @import("std");

const LinearRegressionError = error{
    InsufficientData,
    DimensionMismatch,
};

/// Linear Regression model using gradient descent optimization.
/// This implementation finds the best-fit line for a dataset by iteratively
/// adjusting weights to minimize the sum of squared errors.
pub fn LinearRegression(comptime T: type) type {
    return struct {
        weights: []T,
        allocator: std.mem.Allocator,

        const Self = @This();

        /// Initialize a linear regression model with given number of features
        pub fn init(allocator: std.mem.Allocator, num_features: usize) !Self {
            const weights = try allocator.alloc(T, num_features);
            @memset(weights, 0);
            return Self{
                .weights = weights,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.weights);
        }

        /// Train the model using gradient descent
        /// data_x: input features (each row is a sample, first column should be 1s for bias)
        /// data_y: target values
        /// learning_rate: step size for gradient descent
        /// iterations: number of training iterations
        pub fn fit(
            self: *Self,
            data_x: []const []const T,
            data_y: []const T,
            learning_rate: T,
            iterations: usize,
        ) !void {
            if (data_x.len == 0 or data_y.len == 0) {
                return LinearRegressionError.InsufficientData;
            }
            if (data_x.len != data_y.len) {
                return LinearRegressionError.DimensionMismatch;
            }
            if (data_x[0].len != self.weights.len) {
                return LinearRegressionError.DimensionMismatch;
            }

            const n: T = @floatFromInt(data_x.len);

            var iter: usize = 0;
            while (iter < iterations) : (iter += 1) {
                // Calculate gradient for each weight
                var gradients = try self.allocator.alloc(T, self.weights.len);
                defer self.allocator.free(gradients);
                @memset(gradients, 0);

                // For each training example
                for (data_x, data_y) |x, y| {
                    const prediction = self.predictSingle(x);
                    const error_val = prediction - y;

                    // Update gradient for each feature
                    for (x, 0..) |feature, j| {
                        gradients[j] += error_val * feature;
                    }
                }

                // Update weights using gradient descent
                for (self.weights, gradients) |*weight, gradient| {
                    weight.* -= (learning_rate / n) * gradient;
                }
            }
        }

        /// Predict a single value given features
        fn predictSingle(self: *const Self, features: []const T) T {
            var result: T = 0;
            for (self.weights, features) |weight, feature| {
                result += weight * feature;
            }
            return result;
        }

        /// Predict values for multiple samples
        pub fn predict(self: *const Self, allocator: std.mem.Allocator, data_x: []const []const T) ![]T {
            const predictions = try allocator.alloc(T, data_x.len);
            for (data_x, 0..) |x, i| {
                predictions[i] = self.predictSingle(x);
            }
            return predictions;
        }

        /// Calculate mean squared error
        pub fn meanSquaredError(self: *const Self, data_x: []const []const T, data_y: []const T) T {
            var total_error: T = 0;
            for (data_x, data_y) |x, y| {
                const prediction = self.predictSingle(x);
                const error_val = prediction - y;
                total_error += error_val * error_val;
            }
            const n: T = @floatFromInt(data_x.len);
            return total_error / n;
        }

        /// Calculate mean absolute error
        pub fn meanAbsoluteError(self: *const Self, data_x: []const []const T, data_y: []const T) T {
            var total_error: T = 0;
            for (data_x, data_y) |x, y| {
                const prediction = self.predictSingle(x);
                const error_val = prediction - y;
                total_error += @abs(error_val);
            }
            const n: T = @floatFromInt(data_x.len);
            return total_error / n;
        }
    };
}

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

test "LinearRegression - simple line y = 2x + 1" {
    const allocator = testing.allocator;

    // Create training data: y = 2x + 1
    const x_data = [_][2]f32{
        .{ 1.0, 0.0 }, // bias, x
        .{ 1.0, 1.0 },
        .{ 1.0, 2.0 },
        .{ 1.0, 3.0 },
        .{ 1.0, 4.0 },
    };
    const y_data = [_]f32{ 1.0, 3.0, 5.0, 7.0, 9.0 };

    var x_ptrs: [x_data.len][]const f32 = undefined;
    for (&x_data, 0..) |*row, i| {
        x_ptrs[i] = row;
    }

    var model = try LinearRegression(f32).init(allocator, 2);
    defer model.deinit();

    try model.fit(&x_ptrs, &y_data, 0.01, 1000);

    // Check that weights are close to [1.0, 2.0] (bias, slope)
    try expectApproxEqAbs(@as(f32, 1.0), model.weights[0], 0.1);
    try expectApproxEqAbs(@as(f32, 2.0), model.weights[1], 0.1);

    // Test prediction
    const test_x = [_]f32{ 1.0, 5.0 };
    const pred = model.predictSingle(&test_x);
    try expectApproxEqAbs(@as(f32, 11.0), pred, 0.5);
}

test "LinearRegression - horizontal line y = 3" {
    const allocator = testing.allocator;

    // Create training data: y = 3 (constant)
    const x_data = [_][1]f32{
        .{1.0},
        .{1.0},
        .{1.0},
        .{1.0},
    };
    const y_data = [_]f32{ 3.0, 3.0, 3.0, 3.0 };

    var x_ptrs: [x_data.len][]const f32 = undefined;
    for (&x_data, 0..) |*row, i| {
        x_ptrs[i] = row;
    }

    var model = try LinearRegression(f32).init(allocator, 1);
    defer model.deinit();

    try model.fit(&x_ptrs, &y_data, 0.1, 100);

    try expectApproxEqAbs(@as(f32, 3.0), model.weights[0], 0.1);
}

test "LinearRegression - multiple features" {
    const allocator = testing.allocator;

    // y = 1 + 2*x1 + 3*x2
    const x_data = [_][3]f32{
        .{ 1.0, 1.0, 1.0 },
        .{ 1.0, 2.0, 1.0 },
        .{ 1.0, 1.0, 2.0 },
        .{ 1.0, 2.0, 2.0 },
        .{ 1.0, 3.0, 1.0 },
        .{ 1.0, 1.0, 3.0 },
    };
    const y_data = [_]f32{ 6.0, 8.0, 9.0, 11.0, 10.0, 12.0 };

    var x_ptrs: [x_data.len][]const f32 = undefined;
    for (&x_data, 0..) |*row, i| {
        x_ptrs[i] = row;
    }

    var model = try LinearRegression(f32).init(allocator, 3);
    defer model.deinit();

    try model.fit(&x_ptrs, &y_data, 0.01, 2000);

    // Weights should be approximately [1.0, 2.0, 3.0]
    try expectApproxEqAbs(@as(f32, 1.0), model.weights[0], 0.2);
    try expectApproxEqAbs(@as(f32, 2.0), model.weights[1], 0.2);
    try expectApproxEqAbs(@as(f32, 3.0), model.weights[2], 0.2);
}

test "LinearRegression - predict multiple samples" {
    const allocator = testing.allocator;

    const x_data = [_][2]f32{
        .{ 1.0, 1.0 },
        .{ 1.0, 2.0 },
        .{ 1.0, 3.0 },
    };
    const y_data = [_]f32{ 3.0, 5.0, 7.0 };

    var x_ptrs: [x_data.len][]const f32 = undefined;
    for (&x_data, 0..) |*row, i| {
        x_ptrs[i] = row;
    }

    var model = try LinearRegression(f32).init(allocator, 2);
    defer model.deinit();

    try model.fit(&x_ptrs, &y_data, 0.01, 1000);

    const test_data = [_][2]f32{
        .{ 1.0, 4.0 },
        .{ 1.0, 5.0 },
    };
    var test_ptrs: [test_data.len][]const f32 = undefined;
    for (&test_data, 0..) |*row, i| {
        test_ptrs[i] = row;
    }

    const predictions = try model.predict(allocator, &test_ptrs);
    defer allocator.free(predictions);

    try expectApproxEqAbs(@as(f32, 9.0), predictions[0], 0.5);
    try expectApproxEqAbs(@as(f32, 11.0), predictions[1], 0.5);
}

test "LinearRegression - mean squared error" {
    const allocator = testing.allocator;

    const x_data = [_][2]f32{
        .{ 1.0, 1.0 },
        .{ 1.0, 2.0 },
        .{ 1.0, 3.0 },
    };
    const y_data = [_]f32{ 2.0, 4.0, 6.0 };

    var x_ptrs: [x_data.len][]const f32 = undefined;
    for (&x_data, 0..) |*row, i| {
        x_ptrs[i] = row;
    }

    var model = try LinearRegression(f32).init(allocator, 2);
    defer model.deinit();

    const initial_mse = model.meanSquaredError(&x_ptrs, &y_data);

    try model.fit(&x_ptrs, &y_data, 0.01, 1000);

    const final_mse = model.meanSquaredError(&x_ptrs, &y_data);

    // MSE should decrease after training
    try testing.expect(final_mse < initial_mse);
    try testing.expect(final_mse < 0.1);
}

test "LinearRegression - mean absolute error" {
    const allocator = testing.allocator;

    const x_data = [_][2]f32{
        .{ 1.0, 1.0 },
        .{ 1.0, 2.0 },
    };
    const y_data = [_]f32{ 3.0, 5.0 };

    var x_ptrs: [x_data.len][]const f32 = undefined;
    for (&x_data, 0..) |*row, i| {
        x_ptrs[i] = row;
    }

    var model = try LinearRegression(f32).init(allocator, 2);
    defer model.deinit();

    try model.fit(&x_ptrs, &y_data, 0.01, 1000);

    const mae = model.meanAbsoluteError(&x_ptrs, &y_data);
    try testing.expect(mae < 0.2);
}

test "LinearRegression - dimension mismatch error" {
    const allocator = testing.allocator;

    const x_data = [_][2]f32{
        .{ 1.0, 1.0 },
        .{ 1.0, 2.0 },
    };
    const y_data = [_]f32{ 3.0, 5.0, 7.0 }; // Wrong size

    var x_ptrs: [x_data.len][]const f32 = undefined;
    for (&x_data, 0..) |*row, i| {
        x_ptrs[i] = row;
    }

    var model = try LinearRegression(f32).init(allocator, 2);
    defer model.deinit();

    try expectEqual(
        LinearRegressionError.DimensionMismatch,
        model.fit(&x_ptrs, &y_data, 0.01, 100),
    );
}

test "LinearRegression - insufficient data error" {
    const allocator = testing.allocator;

    const y_data = [_]f32{};

    var x_ptrs: [0][]const f32 = undefined;

    var model = try LinearRegression(f32).init(allocator, 2);
    defer model.deinit();

    try expectEqual(
        LinearRegressionError.InsufficientData,
        model.fit(&x_ptrs, &y_data, 0.01, 100),
    );
}

test "LinearRegression - works with f64" {
    const allocator = testing.allocator;

    const x_data = [_][2]f64{
        .{ 1.0, 1.0 },
        .{ 1.0, 2.0 },
        .{ 1.0, 3.0 },
    };
    const y_data = [_]f64{ 2.5, 4.5, 6.5 };

    var x_ptrs: [x_data.len][]const f64 = undefined;
    for (&x_data, 0..) |*row, i| {
        x_ptrs[i] = row;
    }

    var model = try LinearRegression(f64).init(allocator, 2);
    defer model.deinit();

    try model.fit(&x_ptrs, &y_data, 0.01, 1000);

    try expectApproxEqAbs(@as(f64, 0.5), model.weights[0], 0.1);
    try expectApproxEqAbs(@as(f64, 2.0), model.weights[1], 0.1);
}
