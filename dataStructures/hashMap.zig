const std = @import("std");
const testing = std.testing;

const errors = error{ KeyNotFound, OutOfMemory };

// Returns a hash map instance using separate chaining for collision resolution.
// Arguments:
//      K: the type of the key (i.e. []const u8, i32, etc...)
//      V: the type of the value (i.e. i32, []const u8, etc...)
pub fn HashMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        // Node struct for the linked list in each bucket
        pub const Node = struct {
            key: K,
            value: V,
            next: ?*Node = null,
        };

        // Bucket is a linked list of nodes
        pub const Bucket = struct {
            head: ?*Node = null,
        };

        allocator: std.mem.Allocator,
        buckets: []Bucket,
        size: usize = 0,
        capacity: usize,

        // Initialize a new hash map with the given capacity
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            const buckets = try allocator.alloc(Bucket, capacity);
            for (buckets) |*bucket| {
                bucket.* = Bucket{};
            }
            return Self{
                .allocator = allocator,
                .buckets = buckets,
                .capacity = capacity,
            };
        }

        // Hash function for integers
        fn hashInt(self: *const Self, key: anytype) usize {
            const hash_value = @as(usize, @intCast(@abs(key)));
            return hash_value % self.capacity;
        }

        // Hash function for strings
        fn hashString(self: *const Self, key: []const u8) usize {
            var hash_value: u32 = 5381;
            for (key) |c| {
                hash_value = ((hash_value << 5) +% hash_value) +% c;
            }
            return @as(usize, hash_value) % self.capacity;
        }

        // Generic hash function dispatcher
        fn hashFunc(self: *const Self, key: K) usize {
            const type_info = @typeInfo(K);
            return switch (type_info) {
                .int => self.hashInt(key),
                .pointer => |ptr_info| {
                    if (ptr_info.size == .slice and ptr_info.child == u8) {
                        return self.hashString(key);
                    }
                    @compileError("Unsupported key type");
                },
                else => @compileError("Unsupported key type"),
            };
        }

        // Compare keys based on type
        fn keysEqual(k1: K, k2: K) bool {
            const type_info = @typeInfo(K);
            return switch (type_info) {
                .int => k1 == k2,
                .pointer => |ptr_info| {
                    if (ptr_info.size == .slice and ptr_info.child == u8) {
                        return std.mem.eql(u8, k1, k2);
                    }
                    return k1 == k2;
                },
                else => k1 == k2,
            };
        }

        // Insert or update a key-value pair
        // Runs in O(1) average case, O(n) worst case
        pub fn put(self: *Self, key: K, value: V) !void {
            const index = self.hashFunc(key);
            var bucket = &self.buckets[index];

            // Check if key already exists
            var current = bucket.head;
            while (current) |node| {
                if (keysEqual(node.key, key)) {
                    node.value = value;
                    return;
                }
                current = node.next;
            }

            // Create new node and add to front of list
            const new_node = try self.allocator.create(Node);
            new_node.* = Node{
                .key = key,
                .value = value,
                .next = bucket.head,
            };
            bucket.head = new_node;
            self.size += 1;
        }

        // Get a value by key
        // Runs in O(1) average case, O(n) worst case
        // Returns KeyNotFound error if key doesn't exist
        pub fn get(self: *const Self, key: K) errors!V {
            const index = self.hashFunc(key);
            const bucket = &self.buckets[index];

            var current = bucket.head;
            while (current) |node| {
                if (keysEqual(node.key, key)) {
                    return node.value;
                }
                current = node.next;
            }

            return errors.KeyNotFound;
        }

        // Remove a key-value pair
        // Runs in O(1) average case, O(n) worst case
        // Returns KeyNotFound error if key doesn't exist
        pub fn remove(self: *Self, key: K) errors!void {
            const index = self.hashFunc(key);
            var bucket = &self.buckets[index];

            var current = bucket.head;
            var prev: ?*Node = null;

            while (current) |node| {
                if (keysEqual(node.key, key)) {
                    if (prev) |p| {
                        p.next = node.next;
                    } else {
                        bucket.head = node.next;
                    }
                    self.allocator.destroy(node);
                    self.size -= 1;
                    return;
                }
                prev = node;
                current = node.next;
            }

            return errors.KeyNotFound;
        }

        // Check if a key exists
        // Runs in O(1) average case, O(n) worst case
        pub fn contains(self: *const Self, key: K) bool {
            const index = self.hashFunc(key);
            const bucket = &self.buckets[index];

            var current = bucket.head;
            while (current) |node| {
                if (keysEqual(node.key, key)) {
                    return true;
                }
                current = node.next;
            }

            return false;
        }

        // Destroy the hash map and free all allocated memory
        pub fn deinit(self: *Self) void {
            for (self.buckets) |*bucket| {
                var current = bucket.head;
                while (current) |node| {
                    const next = node.next;
                    self.allocator.destroy(node);
                    current = next;
                }
            }
            self.allocator.free(self.buckets);
        }
    };
}

test "Testing basic put and get operations with integers" {
    const allocator = std.testing.allocator;

    var map = try HashMap(i32, i32).init(allocator, 10);
    defer map.deinit();

    try map.put(1, 10);
    try map.put(2, 20);
    try map.put(3, 30);

    try testing.expect(try map.get(1) == 10);
    try testing.expect(try map.get(2) == 20);
    try testing.expect(try map.get(3) == 30);
    try testing.expect(map.size == 3);
}

test "Testing update existing key" {
    const allocator = std.testing.allocator;

    var map = try HashMap(i32, i32).init(allocator, 10);
    defer map.deinit();

    try map.put(1, 10);
    try testing.expect(try map.get(1) == 10);

    try map.put(1, 100);
    try testing.expect(try map.get(1) == 100);
    try testing.expect(map.size == 1);
}

test "Testing collision handling" {
    const allocator = std.testing.allocator;

    var map = try HashMap(i32, i32).init(allocator, 5);
    defer map.deinit();

    // These keys will have same hash value (0 % 5 = 0, 5 % 5 = 0, 10 % 5 = 0)
    try map.put(0, 100);
    try map.put(5, 200);
    try map.put(10, 300);

    try testing.expect(try map.get(0) == 100);
    try testing.expect(try map.get(5) == 200);
    try testing.expect(try map.get(10) == 300);
    try testing.expect(map.size == 3);
}

test "Testing remove operation" {
    const allocator = std.testing.allocator;

    var map = try HashMap(i32, i32).init(allocator, 10);
    defer map.deinit();

    try map.put(1, 10);
    try map.put(2, 20);
    try map.put(3, 30);

    try testing.expect(map.size == 3);
    try map.remove(2);
    try testing.expect(map.size == 2);

    _ = map.get(2) catch |err| {
        try testing.expect(err == errors.KeyNotFound);
    };

    try testing.expect(try map.get(1) == 10);
    try testing.expect(try map.get(3) == 30);
}

test "Testing contains operation" {
    const allocator = std.testing.allocator;

    var map = try HashMap(i32, i32).init(allocator, 10);
    defer map.deinit();

    try map.put(1, 10);
    try map.put(2, 20);

    try testing.expect(map.contains(1) == true);
    try testing.expect(map.contains(2) == true);
    try testing.expect(map.contains(3) == false);

    try map.remove(1);
    try testing.expect(map.contains(1) == false);
}

test "Testing with string keys" {
    const allocator = std.testing.allocator;

    var map = try HashMap([]const u8, i32).init(allocator, 10);
    defer map.deinit();

    try map.put("hello", 1);
    try map.put("world", 2);
    try map.put("zig", 3);

    try testing.expect(try map.get("hello") == 1);
    try testing.expect(try map.get("world") == 2);
    try testing.expect(try map.get("zig") == 3);
    try testing.expect(map.size == 3);
}

test "Testing error handling for non-existent keys" {
    const allocator = std.testing.allocator;

    var map = try HashMap(i32, i32).init(allocator, 10);
    defer map.deinit();

    try map.put(1, 10);

    _ = map.get(999) catch |err| {
        try testing.expect(err == errors.KeyNotFound);
    };

    _ = map.remove(999) catch |err| {
        try testing.expect(err == errors.KeyNotFound);
    };
}

test "Testing with different value types" {
    const allocator = std.testing.allocator;

    var map = try HashMap(i32, []const u8).init(allocator, 10);
    defer map.deinit();

    try map.put(1, "first");
    try map.put(2, "second");
    try map.put(3, "third");

    const val1 = try map.get(1);
    const val2 = try map.get(2);
    const val3 = try map.get(3);

    try testing.expect(std.mem.eql(u8, val1, "first"));
    try testing.expect(std.mem.eql(u8, val2, "second"));
    try testing.expect(std.mem.eql(u8, val3, "third"));
}

test "Testing negative integers" {
    const allocator = std.testing.allocator;

    var map = try HashMap(i32, i32).init(allocator, 10);
    defer map.deinit();

    try map.put(-5, 50);
    try map.put(-10, 100);
    try map.put(5, 500);

    try testing.expect(try map.get(-5) == 50);
    try testing.expect(try map.get(-10) == 100);
    try testing.expect(try map.get(5) == 500);
}
