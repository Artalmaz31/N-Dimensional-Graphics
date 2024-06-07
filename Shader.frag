#version 130
#extension GL_ARB_arrays_of_arrays : enable

const int MAX_DIM = 6;
const int MAX_REF = 6;
const int MAX_FACES = 100;
const int MAX_STEPS = 100;
const float MAX_DIST = 100000.0;
const float PI = 3.1415926;

uniform vec2 u_resolution;

uniform vec2 u_mouse;
uniform float u_time;

uniform vec3 u_pos;
uniform float[MAX_DIM - 3] u_dir;

uniform vec2 u_seed1;
uniform vec2 u_seed2;

uvec4 R_STATE;

uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
	uint b = (((z << S1) ^ z) >> S2);
	return (((z & M) << S3) ^ b);	
}

uint LCGStep(uint z, uint A, uint C)
{
	return (A * z + C);	
}

vec2 hash22(vec2 p)
{
	p += u_seed1.x;
	vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yzx+33.33);
	return fract((p3.xx+p3.yz)*p3.zy);
}

float random()
{
	R_STATE.x = TausStep(R_STATE.x, 13, 19, 12, uint(4294967294));
	R_STATE.y = TausStep(R_STATE.y, 2, 25, 4, uint(4294967288));
	R_STATE.z = TausStep(R_STATE.z, 3, 11, 17, uint(4294967280));
	R_STATE.w = LCGStep(R_STATE.w, uint(1664525), uint(1013904223));
	return 2.3283064365387e-10 * float((R_STATE.x ^ R_STATE.y ^ R_STATE.z ^ R_STATE.w));
}

mat2 rot(float a)
{
	float s = sin(a);
	float c = cos(a);
	return mat2(c, -s, s, c);
}

struct Matrix
{
	int n;
	float[MAX_DIM][MAX_DIM] data;
};


Matrix EmptyNSizeMatrix(int n)
{
	float[MAX_DIM][MAX_DIM] arr;

	for (int i = 0; i < MAX_DIM; i++)
	{
		for (int j = 0; j < MAX_DIM; j++)
		{
			arr[i][j] = 0.0;
		}
	}

	return Matrix(n, arr);
}

float Determinant(in Matrix a)
{
    float det = 1.0;
    for (int i = 0; i < a.n; i++)
	{
        int pivot = i;
        for (int j = i + 1; j < a.n; j++)
		{
            if (abs(a.data[j][i]) > abs(a.data[pivot][i]))
			{
                pivot = j;
            }
        }
        if (pivot != i)
		{
			float[MAX_DIM] tmp = a.data[i];
			a.data[i] = a.data[pivot];
			a.data[pivot] = tmp;
            det *= -1;
        }

        if (a.data[i][i] == 0) return 0;
        det *= a.data[i][i];

        for (int j = i + 1; j < a.n; j++)
		{
            float factor = a.data[j][i] / a.data[i][i];
            for (int k = i + 1; k < a.n; k++)
			{
                a.data[j][k] -= factor * a.data[i][k];
            }
        }
    }
    return det;
}

Matrix RotationMatrix(float angle, int a, int b)
{
	Matrix m = EmptyNSizeMatrix(MAX_DIM);
	for (int i = 0; i < MAX_DIM; i++) { m.data[i][i] = 1; }

	m.data[a][a] = cos(angle);
	m.data[a][b] = -sin(angle);
	m.data[b][a] = sin(angle);
	m.data[b][b] = cos(angle);

	return m;
}

Matrix MatrixProduct(Matrix a, Matrix b)
{
	Matrix res = EmptyNSizeMatrix(MAX_DIM);

    for (int i = 0; i < MAX_DIM; i++)
	{
        for (int j = 0; j < MAX_DIM; j++)
		{
            for (int k = 0; k < MAX_DIM; k++)
			{
                res.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }

	return res;
}

struct Ray
{
	float[MAX_DIM] org;
	float[MAX_DIM] dir;
};

struct Material
{
  float glow;
  float refl_prob;
  float refr_coef;
  vec3 color;
};

float[MAX_DIM] ToNVector(float a)
{
	float[MAX_DIM] nv;
	for (int i = 0; i < MAX_DIM; i++) { nv[i] = a; }
	return nv;
}

float[MAX_DIM] ToNVector(in vec2 v)
{
	float[MAX_DIM] nv;
	for (int i = 2; i < MAX_DIM; i++) { nv[i] = 0; }
	nv[0] = v.x; nv[1] = v.y;
	return nv;
}

float[MAX_DIM] ToNVector(in vec3 v)
{
	float[MAX_DIM] nv;
	for (int i = 3; i < MAX_DIM; i++) { nv[i] = 0; }
	nv[0] = v.x; nv[1] = v.y; nv[2] = v.z;
	return nv;
}

float[MAX_DIM] ToNVector(in vec4 v)
{
	float[MAX_DIM] nv;
	for (int i = 4; i < MAX_DIM; i++) { nv[i] = 0; }
	nv[0] = v.x; nv[1] = v.y; nv[2] = v.z, nv[3] = v.w;
	return nv;
}

float[MAX_DIM + 1] ExpandNVector(in float[MAX_DIM] nv, float e)
{
	float[MAX_DIM + 1] env;
	for (int i = 0; i < MAX_DIM; i++) { env[i] = nv[i]; }
	env[MAX_DIM] = e;
	return env;
}

float NVectorLength(in float[MAX_DIM] vec)
{
	float cnt = 0;

	for (int i = 0; i < MAX_DIM; i++)
	{
		cnt += vec[i] * vec[i];
	}

	return sqrt(cnt);
}

float NVectorDistance(in float[MAX_DIM] a, in float[MAX_DIM] b)
{
	float[MAX_DIM] v;

	for (int i = 0; i < MAX_DIM; i++)
	{
		v[i] = a[i] - b[i];
	}

	return NVectorLength(v);
}

float[MAX_DIM] NormalizeNVector(in float[MAX_DIM] v)
{
	float len = NVectorLength(v);
	float[MAX_DIM] new_dir;

	for (int i = 0; i < MAX_DIM; i++)
	{
		new_dir[i] = v[i] / len;
	}
	return new_dir;
}

float[MAX_DIM] InverseNVector(in float[MAX_DIM] v)
{
	float[MAX_DIM] inv;

	for (int i = 0; i < MAX_DIM; i++)
	{
		inv[i] = -v[i];
	}

	return inv;
}

float NVectorDot(in float[MAX_DIM] a, in float[MAX_DIM] b)
{
	float dotProduct = 0;

	for (int i = 0; i < MAX_DIM; i++)
	{
		dotProduct += a[i] * b[i];
	}

	return dotProduct;
}

Matrix GetSubMatrix(in Matrix src, int rowToRemove, int colToRemove)
{
    Matrix subMatrix = EmptyNSizeMatrix(src.n - 1);
    int rowIndex = 0;
    int colIndex = 0;

    for (int i = 0; i < src.n; i++)
    {
        if (i == rowToRemove)
        {
            continue;
        }

        colIndex = 0;

        for (int j = 0; j < src.n; j++)
        {
            if (j == colToRemove)
            {
                continue;
            }

            subMatrix.data[rowIndex][colIndex] = src.data[i][j];
            colIndex++;
        }

        rowIndex++;
    }

    return subMatrix;
}

float[MAX_DIM] NVectorCross(in Matrix matrix)
{
	float[MAX_DIM] cross_vector;
	bool s = false;

	for (int i = 0; i < MAX_DIM; i++)
	{
		Matrix m = GetSubMatrix(matrix, MAX_DIM - 1, i);
		float det = Determinant(m) * (s ? -1 : 1);
		cross_vector[i] = det;
		s = !s;
	}

	return cross_vector;
}


float[MAX_DIM] NVectorMix(in float[MAX_DIM] a, in float[MAX_DIM] b, float t)
{
	float[MAX_DIM] res;

	for (int i = 0; i < MAX_DIM; i++)
	{
		res[i] = a[i] + (b[i] - a[i]) * t;
	}

	return res;
}

float[MAX_DIM] ReflectNVector(in float[MAX_DIM] d, in float[MAX_DIM] n)
{
	float dt = NVectorDot(d, n);
	float[MAX_DIM] res;

	for (int i = 0; i < MAX_DIM; i++)
	{
		res[i] = n[i] * dt * (-2) + d[i];
	}

	return NormalizeNVector(res);
}

float[MAX_DIM] RefractNVector(in float[MAX_DIM] d, in float[MAX_DIM] n, float env_refr_coef, float mat_refr_coef)
{
    float eta = env_refr_coef / mat_refr_coef;
    float cos_theta = -NVectorDot(n, d);

    if (cos_theta < 0)
    {
		cos_theta *= -1.0;
		n = InverseNVector(n);
		eta = 1.0 / eta;
    }

    float k = 1.0 - eta * eta * (1.0 - cos_theta * cos_theta);

    if(k >= 0)
	{
		float[MAX_DIM] res;
		for (int i = 0; i < MAX_DIM; i++)
		{
			res[i] = eta * d[i] + (eta * cos_theta - sqrt(k)) * n[i];
		}
		return NormalizeNVector(res);
	}

	return NormalizeNVector(d);
}

float[MAX_DIM] RotateNVector(in float[MAX_DIM] vector, in Matrix matrix)
{
	float[MAX_DIM] oriented_vector;

	for (int i = 0; i < MAX_DIM; i++)
	{
		float x = 0;
		for (int j = 0; j < MAX_DIM; j++)
		{
			x += matrix.data[i][j] * vector[j];
		}
		oriented_vector[i] = x;
	}

	return oriented_vector;
}

float NVectorAngle(in float[MAX_DIM] a, in float[MAX_DIM] b)
{
	float angle_cos = NVectorDot(a, b) / (NVectorLength(a) * NVectorLength(b));
	return acos(angle_cos);
}

struct Sun
{
	float[MAX_DIM] dir;
	float angular_size;
	vec3 light;
};

const vec3 sky_color = vec3(1.2, 2.4, 4.0);
Sun sun = Sun(ToNVector(vec3(-1)), PI * 0.02, vec3(4000, 2400, 1200));

vec3 FinalLight(in float[MAX_DIM] dir)
{
	float deviation = NVectorAngle(dir, sun.dir);
	if (deviation < sun.angular_size)
	{
		float k = deviation / sun.angular_size;
		return sun.light * (1 - k) + sky_color * k;
	}
	else return sky_color;
}

struct Intersection
{
	bool did_intersect;
	float dist;
	float[MAX_DIM] norm;
	Material material;
};

Intersection NULL_INTERSECTION = Intersection(false, 0, ToNVector(vec2(0)), Material(0, 0, 1, vec3(0)));

struct NPlane
{
	float[MAX_DIM + 1] equation_coefs;
	Material material;
};

struct NSphere
{
	float[MAX_DIM] center;
	float radius;
	Material material;
};

struct NPolyhedron
{
	int faces_count;
	float[MAX_FACES][MAX_DIM] faces_normals;
	float[MAX_FACES] faces_offsets;
	Material material;
};

struct NBox
{
	float[MAX_DIM] center;
	float[MAX_DIM] size;
	Material material;
};

struct NBoxOriented
{
	float[MAX_DIM] center;
	float[MAX_DIM] size;
	Matrix orientation;
	Material material;
};

NPolyhedron RotatePolyhedron(in NPolyhedron polyhedron, in Matrix rotmatrix)
{
	for (int i = 0; i < polyhedron.faces_count; i++)
	{
		polyhedron.faces_normals[i] = RotateNVector(polyhedron.faces_normals[i], rotmatrix);
	}

	return polyhedron;
}

float[MAX_DIM] NPlaneNormal(in NPlane p)
{
	float[MAX_DIM] n;

	for (int i = 0; i < MAX_DIM; i++)
	{
		n[i] = p.equation_coefs[i];
	}

	return NormalizeNVector(n);
}

Intersection NPlaneIntersection(in Ray ray, in NPlane plane)
{
	float a = 0, b = -plane.equation_coefs[MAX_DIM];

	for (int i = 0; i < MAX_DIM; i++)
	{
		a += plane.equation_coefs[i] * ray.dir[i];
		b -= plane.equation_coefs[i] * ray.org[i];
	}

	float t = b / a;
	if (t < 0 || t > MAX_DIST) return NULL_INTERSECTION;

	float[MAX_DIM] it_pos;
	for (int i = 0; i < MAX_DIM; i++)
	{
		it_pos[i] = ray.org[i] + ray.dir[i] * t;
	}

	Intersection it = Intersection(true, NVectorDistance(ray.org, it_pos), NPlaneNormal(plane), plane.material);
	return it;
}

float[MAX_DIM] NSphereNormal(in NSphere sphere, in float[MAX_DIM] pos)
{
	float[MAX_DIM] norm;

	for (int i = 0; i < MAX_DIM; i++)
	{
		norm[i] = pos[i] - sphere.center[i];
	}

	return NormalizeNVector(norm);
}

Intersection NSphereIntersection(in Ray ray, in NSphere sph)
{
	float a = 0, b = 0, c = 0;

	for (int i = 0; i < MAX_DIM; i++)
	{
		a += ray.dir[i] * ray.dir[i];
		b += 2 * ray.dir[i] * (ray.org[i] - sph.center[i]);
		c += ray.org[i] * ray.org[i] - 2 * ray.org[i] * sph.center[i] + sph.center[i] * sph.center[i];
	}

	float D = b * b - 4 * a * (c - sph.radius * sph.radius);
	if (D < 0) return NULL_INTERSECTION;

	float x1 = (-b - sqrt(D)) / (2 * a);
	float x2 = (-b + sqrt(D)) / (2 * a);

	if (min(x1, x2) < 0 || min(x1, x2) > MAX_DIST) return NULL_INTERSECTION;

	float[MAX_DIM] it_pos;
	for (int i = 0; i < MAX_DIM; i++)
	{
		it_pos[i] = ray.org[i] + ray.dir[i] * min(x1, x2);
	}

	Intersection it = Intersection(true, NVectorDistance(ray.org, it_pos), NSphereNormal(sph, it_pos), sph.material);
	return it;
}

Intersection NPolyhedronIntersection(in Ray ray, in NPolyhedron polyhedron)
{
	Intersection best_it = NULL_INTERSECTION;
	for (int i = 0; i < polyhedron.faces_count; i++)
	{
		float[MAX_DIM + 1] c = ExpandNVector(polyhedron.faces_normals[i], polyhedron.faces_offsets[i]);
		NPlane face = NPlane(c, polyhedron.material);

		Intersection it = NPlaneIntersection(ray, face);
		if (it.did_intersect)
		{
			bool point_is_inside = true;
			for (int j = 0; j < MAX_DIM * 2; j++)
			{
				if (i == j) continue;

				float res = polyhedron.faces_offsets[j];
				for (int k = 0; k < MAX_DIM; k++)
				{
					res += polyhedron.faces_normals[j][k] * (ray.org[k] + ray.dir[k] * it.dist);
				}

				if (res < 0)
				{
					point_is_inside = false;
					break;
				}
			}

			if (point_is_inside && (best_it == NULL_INTERSECTION || it.dist < best_it.dist))
				best_it = it;
		}
	}

	if (NVectorDot(best_it.norm, ray.dir) > 0) best_it.norm = InverseNVector(best_it.norm);
	return best_it;
}

Intersection NBoxIntersection(in Ray ray, in NBox box)
{
	float[MAX_FACES][MAX_DIM] faces_normals;
	float[MAX_FACES] faces_offsets;

	for (int i = 0; i < MAX_FACES; i++)
	{
		faces_normals[i] = ToNVector(0); 
		faces_offsets[i] = 0;
	}

	for (int i = 0; i < MAX_DIM * 2; i++)
	{
		int s =  mod(i, 2) == 0 ? 1 : -1;
		faces_normals[i][i / 2] = s;
		faces_offsets[i] = box.size[i / 2] - box.center[i / 2] * s;
	}

	NPolyhedron p = NPolyhedron(MAX_DIM * 2, faces_normals, faces_offsets, box.material);
	return NPolyhedronIntersection(ray, p);
}

Intersection NBoxOrientedIntersection(in Ray ray, in NBoxOriented box)
{
	float[MAX_FACES][MAX_DIM] faces_normals;
	float[MAX_FACES] faces_offsets;

	for (int i = 0; i < MAX_FACES; i++)
	{
		faces_normals[i] = ToNVector(0); 
		faces_offsets[i] = 0;
	}

	for (int i = 0; i < MAX_DIM * 2; i++)
	{
		int s =  mod(i, 2) == 0 ? 1 : -1;
		faces_normals[i][i / 2] = s;
		faces_normals[i] = RotateNVector(faces_normals[i], box.orientation);
		faces_offsets[i] = box.size[i / 2];
	}

	for (int i = 0; i < MAX_DIM; i++)
	{
		ray.org[i] -= box.center[i];
	}

	NPolyhedron p = NPolyhedron(MAX_DIM * 2, faces_normals, faces_offsets, box.material);
	return NPolyhedronIntersection(ray, p);
}

struct NTorus
{
	float[MAX_DIM] center;
	float[MAX_DIM - 1] size;
	Matrix orientation;
	Material material;
};

float NTorusSignedDistance(float[MAX_DIM] point, NTorus torus)
{
	for (int i = 0; i < MAX_DIM; i++) { point[i] -= torus.center[i]; }
	if (torus.orientation != EmptyNSizeMatrix(MAX_DIM))
		point = RotateNVector(point, torus.orientation);

	vec2 q = vec2(point[0], point[1]);
	for (int i = 0; i < MAX_DIM - 2; i++)
	{
		if (torus.size[i + 1] == 0)
			return length(q) - torus.size[i];

		q = vec2(length(q) - torus.size[i], point[i + 2]);
	}
	return length(q) - torus.size[MAX_DIM - 2];
}

float[MAX_DIM] NTorusNormal(in float[MAX_DIM] point, in NTorus torus, float epsilon)
{
	float[MAX_DIM] normal = ToNVector(0);
	for (int i = 0; i < MAX_DIM; i++)
	{
		float[MAX_DIM] a, b;
		for(int j = 0; j < MAX_DIM; j++) { a[j] = point[j]; b[j] = point[j]; }
		a[i] += epsilon; b[i] -= epsilon;
		normal[i] = (NTorusSignedDistance(a, torus) - NTorusSignedDistance(b, torus)) / (epsilon * 2);
	}
	return NormalizeNVector(normal);
}

Intersection NTorusIntersection(in Ray ray, in NTorus torus)
{
	float dist = 0;
	for (int i = 0; i < MAX_STEPS; i++)
	{
		float[MAX_DIM] point;
		for (int i = 0; i < MAX_DIM; i++) { point[i] = ray.org[i] + ray.dir[i] * dist; }

		float curr_dist = NTorusSignedDistance(point, torus);
		if (abs(curr_dist) < 0.001)
		{
			float[MAX_DIM] norm = NTorusNormal(point, torus, 0.001);
			return Intersection(true, dist, norm, torus.material);
		}
		dist += curr_dist;
	}
	return NULL_INTERSECTION;
}

Intersection FindIntersection(inout Ray ray)
{
	Intersection it = NULL_INTERSECTION;

	NPlane[1] planes;
	planes[0] = NPlane(ExpandNVector(ToNVector(vec3(0, 0, -1)), 2), Material(0, 0.1, 1, vec3(0.8, 0.5, 0.2)));

	for (int i = 0; i < planes.length(); i++)
	{
		Intersection new_it = NPlaneIntersection(ray, planes[i]);
		if (new_it.did_intersect)
		{
			if (!it.did_intersect) it = new_it;
			else if (new_it.dist < it.dist) it = new_it;
		}
	}

	NTorus[1] toruses;
	float[MAX_DIM - 1] torusSize;
	for(int i = 0; i < MAX_DIM - 1; i++) { torusSize[i] = 0; }
	torusSize[0] = 4; torusSize[1] = 1; torusSize[2] = 0.25;
	Matrix torusOrientation = MatrixProduct(MatrixProduct(MatrixProduct(MatrixProduct(MatrixProduct(RotationMatrix(sin(u_time), 0, 1), RotationMatrix(sin(u_time), 0, 2)), RotationMatrix(sin(u_time), 0, 3)), RotationMatrix(sin(u_time), 1, 2)), RotationMatrix(sin(u_time), 1, 3)), RotationMatrix(sin(u_time), 2, 3));
	toruses[0] = NTorus(ToNVector(0), torusSize, torusOrientation, Material(0, 0.99, 1, vec3(1, 0.6, 0.3)));

	for (int i = 0; i < toruses.length(); i++)
	{
		Intersection new_it = NTorusIntersection(ray, toruses[i]);
		if (new_it.did_intersect)
		{
			if (!it.did_intersect) it = new_it;
			else if (new_it.dist < it.dist) it = new_it;
		}
	}

	float[MAX_DIM] nvector = ToNVector(0);

	NSphere[4] spheres;
	spheres[0] = NSphere(ToNVector(vec3(3, 3, 1)), 1, Material(0, 0.1, 1, vec3(1, 0.5, 0.25)));
	spheres[1] = NSphere(ToNVector(vec3(3, -3, 1)), 1, Material(0, 0.1, 2, vec3(0.5, 1, 0.25)));
	spheres[2] = NSphere(ToNVector(vec3(-3, 3, 1)), 1, Material(0, 0.99, 1, vec3(1, 0.5, 0.5)));
	spheres[3] = NSphere(ToNVector(vec3(-3, -3, 1)), 1, Material(100, 0, 1, vec3(0.25, 1, 0.5)));

	for (int i = 0; i < spheres.length(); i++)
	{
		Intersection new_it = NSphereIntersection(ray, spheres[i]);
		if (new_it.did_intersect)
		{
			if (!it.did_intersect) it = new_it;
			else if (new_it.dist < it.dist) it = new_it;
		}
	}

	NBox[1] boxes;
	float[MAX_DIM] size = ToNVector(1);
	boxes[0] = NBox(ToNVector(vec3(0, 0, 1)), size, Material(1, 0, 2, vec3(0.55, 0.05, 0.95)));

	for (int i = 0; i < boxes.length(); i++)
	{
		Intersection new_it = NBoxIntersection(ray, boxes[i]);
		if (new_it.did_intersect)
		{
			if (!it.did_intersect) it = new_it;
			else if (new_it.dist < it.dist) it = new_it;
		}
	}

	NBoxOriented[1] orientedBoxes;
	Matrix orientation = MatrixProduct(MatrixProduct(RotationMatrix(PI / 6, 0, 3), RotationMatrix(PI / 6, 1, 3)), RotationMatrix(PI / 6, 2, 3));
	orientedBoxes[0] = NBoxOriented(ToNVector(vec3(0, 4, 1)), size, orientation, Material(1, 0, 2, vec3(0.75, 0.25, 0.15)));

	for (int i = 0; i < orientedBoxes.length(); i++)
	{
		Intersection new_it = NBoxOrientedIntersection(ray, orientedBoxes[i]);
		if (new_it.did_intersect)
		{
			if (!it.did_intersect) it = new_it;
			else if (new_it.dist < it.dist) it = new_it;
		}
	}

	NPolyhedron[1] polyhedrons;
	float[MAX_FACES][MAX_DIM] normals;
	float[MAX_FACES] offsets;
	for (int i = 0; i < MAX_FACES; i++) { offsets[i] = 1; }

	normals[0] = ToNVector(vec4(1.0 / sqrt(2), 0, 1.0 / sqrt(2), 1));
	normals[1] = ToNVector(vec4(0, 1.0 / sqrt(2), 1.0 / sqrt(2), -1));
	normals[2] = ToNVector(vec4(-1.0 / sqrt(2), 0, 1.0 / sqrt(2), 1));
	normals[3] = ToNVector(vec4(0, -1.0 / sqrt(2), 1.0 / sqrt(2), -1));
	normals[4] = ToNVector(vec4(1.0 / sqrt(2), 0, -1.0 / sqrt(2), 1));
	normals[5] = ToNVector(vec4(0, 1.0 / sqrt(2), -1.0 / sqrt(2), -1));
	normals[6] = ToNVector(vec4(-1.0 / sqrt(2), 0, -1.0 / sqrt(2), 1));
	normals[7] = ToNVector(vec4(0, -1.0 / sqrt(2), -1.0 / sqrt(2), -1));

	polyhedrons[0] = NPolyhedron(8, normals, offsets, Material(0, 0, 1, vec3(0.5, 0.5, 0.5)));
	polyhedrons[0] = RotatePolyhedron(polyhedrons[0], MatrixProduct(MatrixProduct(RotationMatrix(PI / 6, 0, 3), RotationMatrix(PI / 6, 1, 3)), RotationMatrix(PI / 6, 2, 3)));

	for (int i = 0; i < polyhedrons.length(); i++)
	{
		Intersection new_it = NPolyhedronIntersection(ray, polyhedrons[i]);
		if (new_it.did_intersect)
		{
			if (!it.did_intersect) it = new_it;
			else if (new_it.dist < it.dist) it = new_it;
		}
	}

	return it;
}

float[MAX_DIM] RedirectRay(in Ray ray, in float[MAX_DIM] n, float rp)
{
	float[MAX_DIM] reflected = ReflectNVector(ray.dir, n);

	float[MAX_DIM] rnd_dir;
	for (int i = 0; i < MAX_DIM; i++) { rnd_dir[i] = random(); }
	float diff_coef = NVectorDot(rnd_dir, n);

	float[MAX_DIM] diffuse;
	for (int i = 0; i < MAX_DIM; i++) { diffuse[i] = rnd_dir[i] * diff_coef; }
	diffuse = NormalizeNVector(diffuse);

	return NormalizeNVector(NVectorMix(diffuse, reflected, rp));
}

vec3 TraceRay(Ray ray)
{
	vec3 result_light = vec3(0);
	vec3 unabsorbed_light_part = vec3(1);
	Intersection previous_it, it;

	for (int i = 0; i <= MAX_REF; i++)
	{
		previous_it = (i == 0) ? NULL_INTERSECTION : it;
		it = FindIntersection(ray);

		if (!it.did_intersect)
		{
			return result_light + unabsorbed_light_part * FinalLight(ray.dir);
		}

		result_light += it.material.color * it.material.glow * unabsorbed_light_part;
		unabsorbed_light_part *= it.material.color;

		if (it.material.refr_coef > 1)
		{
			float fresnel = 1 - abs(NVectorDot(InverseNVector(ray.dir), it.norm));
			if (random() - it.material.refl_prob < fresnel * fresnel)
			{
				ray.dir = ReflectNVector(ray.dir, it.norm);
			}
			else
			{
				for (int j = 0; j < MAX_DIM; j++)
				{
					ray.org[j] += ray.dir[j] * it.dist - it.norm[j] * 0.001;
				}
				ray.dir = RefractNVector(ray.dir, it.norm, previous_it.material.refr_coef, it.material.refr_coef);
			}
		}
		else
		{
			for (int j = 0; j < MAX_DIM; j++)
			{
				ray.org[j] += ray.dir[j] * it.dist + it.norm[j] * 0.001;
			}
			ray.dir = RedirectRay(ray, it.norm, it.material.refl_prob);
		}
	}

	return result_light;
}

out vec4 myOutputColor;
in vec2 TexCoord;

void main()
{
	vec2 uv = (gl_TexCoord[0].xy - 0.5) * u_resolution / u_resolution.y;
	vec2 uvRes = hash22(uv + 1.0) * u_resolution + u_resolution;

	R_STATE.x = uint(u_seed1.x + uvRes.x);
	R_STATE.y = uint(u_seed1.y + uvRes.x);
	R_STATE.z = uint(u_seed2.x + uvRes.y);
	R_STATE.w = uint(u_seed2.y + uvRes.y);

	vec3 rayOrigin = u_pos;
	vec3 rayDirection = normalize(vec3(1.0, uv));
	rayDirection.zx *= rot(-u_mouse.y);
	rayDirection.xy *= rot(u_mouse.x);

	Ray ray = Ray(ToNVector(rayOrigin), ToNVector(rayDirection));
	ray.dir = NormalizeNVector(ray.dir);

	Matrix dirRot = RotationMatrix(u_dir[0], 0, 3);
	for (int i = 1; i < MAX_DIM - 3; i++)
	{
		dirRot = MatrixProduct(dirRot, RotationMatrix(u_dir[i], 0, i + 3));
	}
	ray.dir = RotateNVector(ray.dir, dirRot);

	vec3 col = vec3(0.0);
	int samples = 1;

	for (int i = 0; i < samples; i++)
	{
		col += TraceRay(ray);
	}

	col /= samples;
	col = 1 - 1 / (col + 1);
	myOutputColor = vec4(col, 1.0);
}