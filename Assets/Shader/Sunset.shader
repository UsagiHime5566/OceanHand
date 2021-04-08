Shader "Unlit/Sunset"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Speed ("Speed", Float) = 1.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _Speed;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            // Created by Pheema - 2017
            // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

            #define M_PI (3.14159265358979)
            #define GRAVITY (9.80665)
            #define EPS (1e-3)
            #define RAYMARCH_CLOUD_ITER (16)
            #define WAVENUM (32)

            static const float kSensorWidth = 36e-3;
            static const float kFocalLength = 18e-3;

            static const float2 kWind = float2(0.0, 1.0);
            static const float kCloudHeight = 100.0;
            static const float kOceanScale = 10.0;

            static const float kCameraSpeed = 10.0;
            static const float kCameraHeight = 1.0;
            static const float kCameraShakeAmp = 0.002;
            static const float kCameraRollNoiseAmp = 0.2;

            struct Ray
            {
                float3 o;
                float3 dir;
            };

            struct HitInfo
            {
                float3 pos;
                float3 normal;
                float dist;
                Ray ray;
            };

            float rand(float2 n) { 
                return frac(sin(dot(n, float2(12.9898, 4.1414))) * 43758.5453);
            }

            float rand(float3 n)
            {
                return frac(sin(dot(n, float3(12.9898, 4.1414, 5.87924))) * 43758.5453);
            }

            float Noise2D(float2 p)
            {
                float2 e = float2(0.0, 1.0);
                float2 mn = floor(p);
                float2 xy = frac(p);
                
                float val = lerp(
                lerp(rand(mn + e.xx), rand(mn + e.yx), xy.x),
                lerp(rand(mn + e.xy), rand(mn + e.yy), xy.x),
                xy.y
                );  
                
                val = val * val * (3.0 - 2.0 * val);
                return val;
            }

            float Noise3D(float3 p)
            {
                float2 e = float2(0.0, 1.0);
                float3 i = floor(p);
                float3 f = frac(p);
                
                float x0 = lerp(rand(i + e.xxx), rand(i + e.yxx), f.x);
                float x1 = lerp(rand(i + e.xyx), rand(i + e.yyx), f.x);
                float x2 = lerp(rand(i + e.xxy), rand(i + e.yxy), f.x);
                float x3 = lerp(rand(i + e.xyy), rand(i + e.yyy), f.x);
                
                float y0 = lerp(x0, x1, f.y);
                float y1 = lerp(x2, x3, f.y);
                
                float val = lerp(y0, y1, f.z);
                
                val = val * val * (3.0 - 2.0 * val);
                return val;
            }

            float SmoothNoise(float3 p)
            {
                float amp = 1.0;
                float freq = 1.0;
                float val = 0.0;
                
                for (int i = 0; i < 4; i++)
                {   
                    amp *= 0.5;
                    val += amp * Noise3D(freq * p - float(i) * 11.7179);
                    freq *= 2.0;
                }
                
                return val;
            }

            float Pow5(float x)
            {
                return (x * x) * (x * x) * x;
            }

            // Schlick approx
            // Ref: https://en.wikipedia.org/wiki/Schlick's_approximation
            float FTerm(float LDotH, float f0)
            {
                return f0 + (1.0 - f0) * Pow5(1.0 - LDotH);
            }

            float OceanHeight(float2 p)
            {    
                float height = 0.0;
                float2 grad = float2(0.0, 0.0);
                float t = _Time.y * _Speed;

                float windNorm = length(kWind);
                float windDir = atan2(kWind.y, kWind.x);

                for (int i = 1; i < WAVENUM; i++)
                {   
                    float rndPhi = windDir + asin(2.0 * rand(float2(0.141 * float(i), 0.1981)) - 1.0);
                    float kNorm = 2.0 * M_PI * float(i) / kOceanScale;
                    float2 kDir = float2(cos(rndPhi), sin(rndPhi)); 
                    float2 k = kNorm * kDir;
                    float l = (windNorm * windNorm) / GRAVITY;
                    float amp = exp(-0.5 / (kNorm * kNorm * l * l)) / (kNorm * kNorm);
                    float omega = sqrt(GRAVITY * kNorm + 0.01 * sin(p.x));
                    float phase = 2.0 * M_PI * rand(float2(0.6814 * float(i), 0.7315));

                    float2 p2 = p;
                    p2 -= amp * k * cos(dot(k, p2) - omega * t + phase);
                    height += amp * sin(dot(k, p2) - omega * t + phase);
                }
                return height;
            }

            float3 OceanNormal(float2 p, float3 camPos)
            {
                float2 e = float2(0, 1.0 * EPS);
                float l = 20.0 * distance(float3(p.x, 0.0, p.y), camPos);
                e.y *= l;
                
                float hx = OceanHeight(p + e.yx) - OceanHeight(p - e.yx);
                float hz = OceanHeight(p + e.xy) - OceanHeight(p - e.xy);
                return normalize(float3(-hx, 2.0 * e.y, -hz));
            }

            HitInfo IntersectOcean(Ray ray) {
                HitInfo hit;
                float3 rayPos = ray.o;
                float dl = rayPos.y / abs(ray.dir.y);
                rayPos += ray.dir * dl;
                hit.pos = rayPos;
                hit.normal = OceanNormal(rayPos.xz, ray.o);
                hit.dist = length(rayPos - ray.o);
                return hit;
            }

            float3 RayMarchCloud(Ray ray, float3 sunDir, float3 bgColor)
            {
                float3 rayPos = ray.o;
                rayPos += ray.dir * (kCloudHeight - rayPos.y) / ray.dir.y;
                
                float dl = 1.0;
                float scatter = 0.0;
                float3 t = bgColor;
                for(int i = 0; i < RAYMARCH_CLOUD_ITER; i++) {
                    rayPos += dl * ray.dir;
                    float dens = SmoothNoise(float3(0.05, 0.001 - 0.001 * _Time.y * _Speed, 0.1) * rayPos - float3(0,0, 0.2 * _Time.y * _Speed)) * 
                    SmoothNoise(float3(0.01, 0.01, 0.01) * rayPos);
                    t -= 0.01 * t * dens * dl;
                    t += 0.02 * dens * dl;
                }
                return t;
            }

            // Environment map
            float3 BGColor(float3 dir, float3 sunDir) {
                float3 color = (0);
                
                color += lerp(
                float3(0.094, 0.2266, 0.3711),
                float3(0.988, 0.6953, 0.3805),
                clamp(0.0, 1.0, dot(sunDir, dir) * dot(sunDir, dir)) * smoothstep(-0.1, 0.1, sunDir.y)
                );
                
                dir.x += 0.01 * sin(312.47 * dir.y + _Time.y * _Speed) * exp(-40.0 * dir.y);
                dir = normalize(dir);
                
                color += smoothstep(0.995, 1.0, dot(sunDir, dir)); 
                return color;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed2 uv = i.uv; uv = 2.0 * uv - 1.0;
                fixed4 fragColor;
                fixed2 iResolution = fixed2(800, 450);
                fixed2 fragCoord = i.uv * iResolution;
                

                uv = i.uv * 2.0 - 1.0;
                float aspect = iResolution.y / iResolution.x;
                
                // Camera settings
                float3 camPos = float3(0, kCameraHeight, -kCameraSpeed * _Time.y * _Speed);
                float3 camDir = float3(kCameraShakeAmp * (rand(float2(_Time.y * _Speed, 0.0)) - 0.5), kCameraShakeAmp * (rand(float2(_Time.y * _Speed, 0.1)) - 0.5), -1);
                
                float3 up = float3(kCameraRollNoiseAmp * (SmoothNoise(float3(0.2 * _Time.y * _Speed, 0.0, 0.0)) - 0.5), 1.0, 0.0);
                
                float3 camForward = normalize(camDir);
                float3 camRight = cross(camForward, up);
                float3 camUp = cross(camRight, camForward);
                
                // Ray
                Ray ray;
                ray.o = camPos;
                ray.dir = normalize(
                kFocalLength * camForward + 
                kSensorWidth * 0.5 * uv.x * camRight + 
                kSensorWidth * 0.5 * aspect * uv.y * camUp
                );
                
                // Controll the height of the sun
                float mouseY = 0;//iMouse.y;
                if (mouseY <= 0.0) mouseY = 0.5 * iResolution.y;
                float3 sunDir = normalize(float3(0, -0.1 + 0.3 * mouseY / iResolution.y, -1));
                
                float3 color = (0);
                HitInfo hit;
                float l = 0.0;
                if (ray.dir.y < 0.0) 
                {
                    // Render an ocean
                    HitInfo hit = IntersectOcean(ray);
                    
                    float3 oceanColor = float3(0.0, 0.2648, 0.4421) * dot(-ray.dir, float3(0, 1, 0));
                    float3 refDir = reflect(ray.dir, hit.normal);
                    refDir.y = abs(refDir.y);
                    l = -camPos.y / ray.dir.y;
                    color = oceanColor + BGColor(refDir, sunDir) * FTerm(dot(refDir, hit.normal), 0.5);
                } 
                else 
                {
                    // Render clouds
                    float3 bgColor = BGColor(ray.dir, sunDir);
                    color += RayMarchCloud(ray, sunDir, bgColor);
                    l = (kCloudHeight - camPos.y) / ray.dir.y;
                }
                
                // Fog
                color = lerp(color, BGColor(ray.dir, sunDir), 1.0 - exp(-0.0001 * l));
                
                // Color grading
                color = smoothstep(0.3, 0.8, color);
                fragColor = float4(color, 1.0);

                return fragColor;
            }
            ENDCG
        }
    }
}
