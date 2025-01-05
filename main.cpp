#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <array>
#include <cassert>

// Define a 3D vector
struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    // Vector subtraction
    Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    // Cross product
    Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    // Dot product
    float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    // float multiplication
    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    // Add vectors
    Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    Vec3 normalize() const {
        float norm = std::sqrt(x * x + y * y + z * z);
        return {x / norm, y / norm, z / norm};
    }

    const float operator[] (const size_t& i) const {
        assert(i < 3);
        if (i == 0) {
            return x;
        } else if (i == 1) {
            return y;
        } else {
            return z;
        }
    }

    float& operator[] (const size_t& i) {
        assert(i < 3);
        if (i == 0) {
            return x;
        } else if (i == 1) {
            return y;
        } else {
            return z;
        }
    }
};

class Triangle {

public:

    Triangle() {}
    Triangle(const Vec3& p00, const Vec3& p11, const Vec3& p22) {
        p0 = p00;
        p1 = p11;
        p2 = p22;
    }

public:
    Vec3 p0;
    Vec3 p1;
    Vec3 p2;

    Vec3 operator[](const size_t i) const {
        assert(i < 3);
        if (i == 0) {
            return p0;
        } else if (i == 1) {
            return p1;
        } else {
            return p2;
        }
    }

    Vec3& operator[](const size_t i) {
        assert(i < 3);
        if (i == 0) {
            return p0;
        } else if (i == 1) {
            return p1;
        } else {
            return p2;
        }
    }
};

std::pair<float, float> project_polygon(const Vec3& axis, const Triangle& triangle) {
    float min_proj = std::numeric_limits<float>::infinity();
    float max_proj = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < 3; ++i) {
        Vec3 point = triangle[i];
        float proj = axis.dot(point);
        min_proj = std::min(min_proj, proj);
        max_proj = std::max(max_proj, proj);
    }
    for (size_t i = 0; i < 3; ++i) {
        Vec3 point = triangle[i];
        float proj = axis.dot(point);
        min_proj = std::min(min_proj, proj);
        max_proj = std::max(max_proj, proj);
    }
    return {min_proj, max_proj};
}

bool polygons_overlap(const std::pair<float, float>& proj1, const std::pair<float, float>& proj2) {
    return proj1.first <= proj2.second && proj2.first <= proj1.second;
}

bool is_coplanar(const Triangle& triangle1, const Triangle& triangle2) {
    Vec3 normal1 = (triangle1[1] - triangle1[0]).cross(triangle1[2] - triangle1[0]).normalize();
    Vec3 normal2 = (triangle2[1] - triangle2[0]).cross(triangle2[2] - triangle2[0]).normalize();

    Vec3 cross_product = normal1.cross(normal2);
    if (std::sqrt(cross_product.x * cross_product.x + cross_product.y * cross_product.y + cross_product.z * cross_product.z) < 1e-6) {
        Vec3 vector_between_triangles = triangle2[0] - triangle1[0];
        if (std::abs(normal1.dot(vector_between_triangles)) < 1e-6) {
            return true;
        }
    }
    return false;
}

// Function to check if two points are equal
bool pointsEqual(const Vec3& p1, const Vec3& p2) {
    return std::fabs(p1[0] - p2[0]) < 1e-6 && std::fabs(p1[1] - p2[1]) < 1e-6 && std::fabs(p1[2] - p2[2]) < 1e-6;
}

// Function to check if two triangles share an edge
bool shareEdge(const Triangle& tri1, const Triangle& tri2) {
    int sharedEdges = 0;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if ((pointsEqual(tri1[i], tri2[j]) && pointsEqual(tri1[(i + 1) % 3], tri2[(j + 1) % 3])) ||
                (pointsEqual(tri1[i], tri2[(j + 1) % 3]) && pointsEqual(tri1[(i + 1) % 3], tri2[j]))) {
                sharedEdges += 1;
            }
        }
    }

    return sharedEdges == 1;
}

// Function to check if two triangles share a point
bool sharePoint(const Triangle& tri1, const Triangle& tri2) {
    for (const auto& p1 : tri1) {
        for (const auto& p2 : tri2) {
            if (pointsEqual(p1, p2)) {
                return true;
            }
        }
    }
    return false;
}

bool triangle_intersection_2d(const Triangle& triangle1, const Triangle& triangle2) {
    std::vector<Vec3> axes;

    for (const auto& tri : {triangle1, triangle2}) {
        for (size_t i = 0; i < 3; ++i) {
            Vec3 edge = tri[(i + 1) % 3] - tri[i];
            Vec3 axis = {edge.y, -edge.x, 0}; // Perpendicular to edge in 2D
            axes.push_back(axis.normalize());
        }
    }

    for (const auto& axis : axes) {
        auto proj1 = project_polygon(axis, triangle1);
        auto proj2 = project_polygon(axis, triangle2);
        if (!polygons_overlap(proj1, proj2)) {
            return false;
        }
    }
    return true;
}

bool triangle_intersection_3d(const Triangle& triangle1, const Triangle& triangle2) {

    if (shareEdge(triangle1, triangle2)) {
        return false;
    }

    std::vector<Vec3> axes;

    for (const auto& tri : {triangle1, triangle2}) {
        Vec3 normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize();
        axes.push_back(normal);
    }

    for (const auto& tri1 : {triangle1, triangle2}) {
        for (const auto& tri2 : {triangle1, triangle2}) {
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    Vec3 edge1 = tri1[(i + 1) % 3] - tri1[i];
                    Vec3 edge2 = tri2[(j + 1) % 3] - tri2[j];
                    Vec3 axis = edge1.cross(edge2).normalize();
                    if (std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z) > 1e-5) {
                        axes.push_back(axis);
                    }
                }
            }
        }
    }

    for (const auto& axis : axes) {
        auto proj1 = project_polygon(axis, triangle1);
        auto proj2 = project_polygon(axis, triangle2);
        if (!polygons_overlap(proj1, proj2)) {
            return false;
        }
    }
    return true;
}

bool triangle_intersection(const Triangle& triangle1, const Triangle& triangle2) {
    if (is_coplanar(triangle1, triangle2)) {
        // Project triangles onto 2D plane by ignoring one coordinate (e.g., z)
        Triangle triangle1_2d(Vec3(triangle1.p0.x, triangle1.p0.y, 0), Vec3(triangle1.p1.x, triangle1.p1.y, 0), Vec3(triangle1.p2.x, triangle1.p2.y, 0));
        Triangle triangle2_2d(Vec3(triangle2.p0.x, triangle2.p0.y, 0), Vec3(triangle2.p1.x, triangle2.p1.y, 0), Vec3(triangle2.p2.x, triangle2.p2.y, 0));
        return triangle_intersection_2d(triangle1_2d, triangle2_2d);
    } else {
        return triangle_intersection_3d(triangle1, triangle2);
    }
}

struct AABB {
    float min[3], max[3];

    // Compute the AABB for a triangle
    void compute(const Triangle& tri) {
        for (int i = 0; i < 3; ++i) {
            min[i] = std::min({tri[0][i], tri[1][i], tri[2][i]});
            max[i] = std::max({tri[0][i], tri[1][i], tri[2][i]});
        }
    }

    // Check if two AABBs intersect
    bool intersects(const AABB& other) const {
        for (int i = 0; i < 3; ++i) {
            if (max[i] < other.min[i] || min[i] > other.max[i]) return false;
        }
        return true;
    }

    std::vector<Vec3> getVertices() const {
        std::vector<Vec3> vertices;
        vertices.resize(8);
        // Compute the 8 vertices
        vertices[0] = {min[0], min[1], min[2]};
        vertices[1] = {max[0], min[1], min[2]};
        vertices[2] = {min[0], max[1], min[2]};
        vertices[3] = {max[0], max[1], min[2]};
        vertices[4] = {min[0], min[1], max[2]};
        vertices[5] = {max[0], min[1], max[2]};
        vertices[6] = {min[0], max[1], max[2]};
        vertices[7] = {max[0], max[1], max[2]};
        return vertices;
    }
};

struct BVHNode {
    AABB box;
    int leftChild;   // Index of left child (-1 if leaf)
    int rightChild;  // Index of right child (-1 if leaf)
    int triangleIndex; // Triangle index if this is a leaf

    BVHNode() : leftChild(-1), rightChild(-1), triangleIndex(-1) {}

    bool isLeaf() const {
        if ((leftChild == -1) && (rightChild == -1)) {
            return true;
        }

        return false;
    }
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<Triangle> triangles;

    int build(int start, int end) {
        BVHNode node;
        // Compute AABB for this node
        for (int i = start; i < end; ++i) {
            AABB triBox;
            triBox.compute(triangles[i]);
            if (i == start) node.box = triBox;
            else {
                for (int j = 0; j < 3; ++j) {
                    node.box.min[j] = std::min(node.box.min[j], triBox.min[j]);
                    node.box.max[j] = std::max(node.box.max[j], triBox.max[j]);
                }
            }
        }

        // If leaf node
        if (end - start == 1) {
            node.triangleIndex = start;
            nodes.push_back(node);
            return nodes.size() - 1;
        }

        // Find split axis
        int axis = 0;
        float maxExtent = 0;
        for (int i = 0; i < 3; ++i) {
            float extent = node.box.max[i] - node.box.min[i];
            if (extent > maxExtent) {
                maxExtent = extent;
                axis = i;
            }
        }

        // Sort triangles along the split axis
        std::sort(triangles.begin() + start, triangles.begin() + end,
                  [axis](const Triangle& a, const Triangle& b) {
            float centerA = (a[0][axis] + a[1][axis] + a[2][axis]) / 3.0f;
            float centerB = (b[0][axis] + b[1][axis] + b[2][axis]) / 3.0f;
            return centerA < centerB;
        });

        // Split and build children
        int mid = (start + end) / 2;
        node.leftChild = build(start, mid);
        node.rightChild = build(mid, end);

        nodes.push_back(node);
        return nodes.size() - 1;
    }

    void initialize(const std::vector<Triangle>& inputTriangles) {
        triangles = inputTriangles;
        nodes.clear();
        build(0, triangles.size());
    }

    void exportCubesToFile(const std::string& filename) {
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
            return;
        }

        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto vertices = nodes[i].box.getVertices();
            for (size_t j = 0; j < 8; ++j) {
                file << i << " " << vertices[j].x << " " << vertices[j].y << " " << vertices[j].z << std::endl;
            }
            file << std::endl;
        }

        file.close();
        std::cout << "BVH exported to " << filename << std::endl;
    }

    void exportPointsToFile(const std::string& filename) {
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
            return;
        }

        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto pMin = nodes[i].box.min;
            const auto pMax = nodes[i].box.max;
            file << i << " " << pMin[0] << " " << pMin[1] << " " << pMin[2] << std::endl;
            file << i << " " << pMax[0] << " " << pMax[1] << " " << pMax[2] << std::endl;
            file << std::endl;
        }
        file.close();
        std::cout << "BVH exported to " << filename << std::endl;
    }

    void exportBVHDataToFile(const std::string& filename) {
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
            return;
        }

        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto pMin = nodes[i].box.min;
            const auto pMax = nodes[i].box.max;
            file << i << " " << pMin[0] << " " << pMin[1] << " " << pMin[2] << std::endl;
            file << i << " " << pMax[0] << " " << pMax[1] << " " << pMax[2] << std::endl;
            file << "Triangle index: " << nodes[i].triangleIndex;
            file << std::endl;
        }
        file.close();
        std::cout << "BVH exported to " << filename << std::endl;
    }
};

// // Check if two ranges overlap
// bool rangesOverlap(float min1, float max1, float min2, float max2) {
//     return !(max1 < min2 || max2 < min1);
// }

// // Cross product of two 3D vectors
// void crossProduct(const Vec3& u, const Vec3& v, Vec3& result) {
//     float result0 = u[1] * v[2] - u[2] * v[1];
//     float result1 = u[2] * v[0] - u[0] * v[2];
//     float result2  = u[0] * v[1] - u[1] * v[0];

//     result.x = result0;
//     result.y = result1;
//     result.z = result2;
// }

// // Dot product of two 3D vectors
// float dotProduct(const Vec3& u, const Vec3& v) {
//     return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
// }

// // Subtract two 3D vectors
// void subtract(const Vec3& u, const Vec3& v, Vec3& result) {
//     result = u - v;
//     // result[0] = u[0] - v[0];
//     // result[1] = u[1] - v[1];
//     // result[2] = u[2] - v[2];
// }

// void projectTriangleOntoAxis(const Triangle& tri, const Vec3& axis, float& min, float& max) {
//     float projection0 = dotProduct(tri[0], axis);
//     float projection1 = dotProduct(tri[1], axis);
//     float projection2 = dotProduct(tri[2], axis);

//     min = std::min({projection0, projection1, projection2});
//     max = std::max({projection0, projection1, projection2});
// }


// bool triangleIntersect(const Triangle& tri1, const Triangle& tri2) {
//     const Vec3 v0 = tri1[0];
//     const Vec3 v1 = tri1[1];
//     const Vec3 v2 = tri1[2];

//     const Vec3 u0 = tri2[0];
//     const Vec3 u1 = tri2[1];
//     const Vec3 u2 = tri2[2];

//     // Compute edges of the triangles
//     Vec3 e1, e2, f1, f2;

//     subtract(v1, v0, e1);
//     subtract(v2, v0, e2);
//     subtract(u1, u0, f1);
//     subtract(u2, u0, f2);

//     // Compute normals of the triangles
//     Vec3 n1, n2;
//     crossProduct(e1, e2, n1);
//     crossProduct(f1, f2, n2);

//     // Test the triangle normals as separating axes
//     float tri1Min, tri1Max, tri2Min, tri2Max;

//     projectTriangleOntoAxis(tri1, n1, tri1Min, tri1Max);
//     projectTriangleOntoAxis(tri2, n1, tri2Min, tri2Max);
//     if (!rangesOverlap(tri1Min, tri1Max, tri2Min, tri2Max)) return false;

//     projectTriangleOntoAxis(tri1, n2, tri1Min, tri1Max);
//     projectTriangleOntoAxis(tri2, n2, tri2Min, tri2Max);
//     if (!rangesOverlap(tri1Min, tri1Max, tri2Min, tri2Max)) return false;

//     // Test cross products of edges as separating axes
//     // float axes[9][3];
//     // crossProduct(e1, f1, axes[0]);
//     // crossProduct(e1, f2, axes[1]);
//     // crossProduct(e2, f1, axes[2]);
//     // crossProduct(e2, f2, axes[3]);

//     Vec3* axesArray = new Vec3[9];
//     crossProduct(e1, f1, axesArray[0]);
//     crossProduct(e1, f2, axesArray[1]);
//     crossProduct(e2, f1, axesArray[2]);
//     crossProduct(e2, f2, axesArray[3]);


//     for (int i = 0; i < 9; ++i) {
//         projectTriangleOntoAxis(tri1, axesArray[i], tri1Min, tri1Max);
//         projectTriangleOntoAxis(tri2, axesArray[i], tri2Min, tri2Max);
//         if (!rangesOverlap(tri1Min, tri1Max, tri2Min, tri2Max)) return false;
//     }

//     // If no separating axis is found, triangles intersect
//     return true;
// }

void detectIntersections(const BVH& bvh, int nodeA, int nodeB, std::vector<std::pair<int, int>>& results) {
    const BVHNode& a = bvh.nodes[nodeA];
    const BVHNode& b = bvh.nodes[nodeB];

    // AABB intersection test
    if (!a.box.intersects(b.box)) return;

    // If both are leaves, test triangles
    if (a.leftChild == -1 && b.leftChild == -1) {

        if (triangle_intersection_3d(bvh.triangles[a.triangleIndex], bvh.triangles[b.triangleIndex])) {
            results.emplace_back(a.triangleIndex, b.triangleIndex);
            const auto t1 = bvh.triangles[a.triangleIndex];
            const auto t2 = bvh.triangles[b.triangleIndex];

            printf("tri1 = np.array([(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)])\n"
                   "tri2 = np.array([(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)])\n",
                   t1.p0.x, t1.p0.y, t1.p0.z, t1.p1.x, t1.p1.y, t1.p1.z, t1.p2.x, t1.p2.y, t1.p2.z,
                   t2.p0.x, t2.p0.y, t2.p0.z, t2.p1.x, t2.p1.y, t2.p1.z, t2.p2.x, t2.p2.y, t2.p2.z);
        };

        return;
    }

    // Recursively traverse children
    if (a.leftChild != -1) {
        detectIntersections(bvh, a.leftChild, nodeB, results);
        detectIntersections(bvh, a.rightChild, nodeB, results);
    } else {
        detectIntersections(bvh, nodeA, b.leftChild, results);
        detectIntersections(bvh, nodeA, b.rightChild, results);
    }
}

void detectAllIntersections(const BVH& bvh, std::vector<std::pair<int, int>>& results) {
    for (int i = 0; i < bvh.nodes.size(); ++i) {
        for (int j = 0; j < bvh.nodes.size(); ++j) {
            if (i == j) continue;
            detectIntersections(bvh, i, j, results);
        }
    }
}

bool loadOBJ(const std::string& filename, std::vector<Triangle>& triangles) {
    std::vector<Vec3> vertices;

    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open OBJ file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string prefix;
        stream >> prefix;

        if (prefix == "v") {
            // Vec3 position
            float x, y, z;
            stream >> x >> y >> z;
            vertices.push_back({x, y, z});
        } else if (prefix == "f") {
            // Face definition
            std::vector<int> vertexIndices;
            std::string vertexInfo;
            while (stream >> vertexInfo) {
                // Parse the vertex index (ignoring texture/normal indices)
                size_t slashPos = vertexInfo.find('/');
                int index = std::stoi(vertexInfo.substr(0, slashPos)) - 1; // OBJ indices are 1-based
                vertexIndices.push_back(index);
            }

            // Triangulate the face if necessary
            for (size_t i = 1; i + 1 < vertexIndices.size(); ++i) {
                Triangle triangle;
                for (int j = 0; j < 3; ++j) {
                    const auto& vertex = vertices[vertexIndices[j == 0 ? 0 : j == 1 ? i : i + 1]];
                    triangle[j][0] = vertex.x;
                    triangle[j][1] = vertex.y;
                    triangle[j][2] = vertex.z;
                }
                triangles.push_back(triangle);
            }
        }
    }

    file.close();
    return true;
}


int main()
{
    Triangle triangle1(Vec3(0.340, 0.639, 0.698), Vec3(0.737, 0.083, 0.159), Vec3(0.744, 0.590, 0.548));
    Triangle triangle2(Vec3(0.271, 0.254, 0.444), Vec3(0.374, 0.442, 0.113), Vec3(0.607, 0.621, 0.887));


    if (triangle_intersection(triangle1, triangle2)) {
        std::cout << "Triangles intersect!" << std::endl;
    } else {
        std::cout << "Triangles do not intersect!" << std::endl;
    }

    std::vector<Triangle> triangles;

    if (!loadOBJ("/home/abdellah/Downloads/irregular.obj", triangles)) {
        std::cout << "Error \n";
    } else {
        std::cout << "Mesh loaded \n";
    }

    // Build BVH
    BVH bvh;
    bvh.initialize(triangles);

    bvh.exportPointsToFile("/home/abdellah/Downloads/irregular.pts");
    bvh.exportBVHDataToFile("/home/abdellah/Downloads/irregular.bvh");


    // Detect self-intersections
    std::vector<std::pair<int, int>> intersections;
    detectAllIntersections(bvh, intersections);



    std::cout << "Number intersections: " << intersections.size() << std::endl;
    // // Output results
    // for (const auto& pair : intersections) {
    //     std::cout << "Intersection between triangles: " << pair.first << " and " << pair.second << std::endl;
    // }

    return 0;
}



