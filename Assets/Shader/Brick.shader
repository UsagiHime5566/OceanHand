Shader "Custom/Brick"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        iChannel0("iChannel0", 2D) = "white" {}
        _Speed ("Speed", Float) = 1.0
        _Color1 ("Color1", Color) = (0.5, 0.5, 0.5, 1)
        _Color2 ("Color2", Color) = (0.67, 0.85, 0.8, 1)
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
            sampler2D iChannel0;
            float _Speed;
            float4  _Color1;
            float4  _Color2;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }
            #define ivec2 int2
            #define ivec3 int3
            #define ivec4 int4
            #define vec2 float2
            #define vec3 float3
            #define vec4 float4
            #define svec2(x) float2(x,x)
            #define svec3(x) float3(x,x,x)
            #define svec4(x) float4(x,x,x,x)

            #define mat2 float2x2
            #define mat3 float3x3
            #define mat4 float4x4
            #define iTime _Time.y
            // fmod用於求余數，比如fmod(1.5, 1.0) 返回0.5；
            #define mod fmod
            // 插值運算，lerp(a,b,w) return a + w*(a-b);
            #define mix lerp
            // fract(x): 取小數部分 
            #define fract frac
            #define atan atan2
            #define texture2D tex2D

            #define _fixed2(x) fixed2(x,x)
            #define _fixed3(x) fixed3(x,x,x)
            #define _fixed4(x) fixed4(x,x,x,x)



            float Hash21(vec2 p){
                p = fract(p*vec2(123.34,456.21));
                p += dot(p, p+45.32);
                return fract(p.x*p.y);
            }

            float rand(vec2 p){
                return mix(.2, .7, Hash21(p));
            }

            fixed4 frag (v2f i) : SV_Target
            {
                //step1. uv 改為 i.uv , 因為貼圖一定是 1 x 1 矩形, 不必像GLSH一樣局限於瀏覽器長寬
                //step2. iGlobalTime 改為 _Time.y
                //step3. mod 改 fmod  or  #define mod(x,y) (x-y*floor(x/y))
                //step4. mix 改 lerp
                //step5. fract 改 frac
                //step6. mat2 矩陣 改為 fixed2x2
                //step7. 矩陣運算 a*b 須改為 mul(a,b)  , 通常發生此問題的編譯錯誤都是 "type mismatch"
                //int2 轉換為 float2  ->  (float2)int2
                //fragCoord.xy -> gl_FragCoord.xy
                //fragColor -> gl_FragColor
                fixed2 uv = i.uv; uv = 2.0 * uv - 1.0;
                fixed4 fragColor;
                fixed2 fragCoord = i.uv * 400;
                fixed2 iResolution = fixed2(400, 400);


                uv = fragCoord / iResolution.y * 10.;
    
                uv.x += _Speed * iTime / 2.;
                uv.y -= 10.;

                vec2 f = fract(uv);
                vec2 c = floor(uv);
                
                vec2 texCoord = f;
                vec2 tileCoord = c;
                
                float r0 = rand(floor(uv));
                
                int orientation = (int(c.x) + int(c.y)) & 1;
                
                //if(f[orientation] > r0)
                //    tileCoord[orientation] += 1.;
                if(f[orientation] > r0){
                    if(orientation == 0)
                        tileCoord.x += 1.;
                    else if(orientation == 1)
                        tileCoord.y += 1.;
                }

                //texCoord[orientation] = abs(texCoord[orientation] - r0);
                if(orientation == 0)
                    texCoord.x = abs(texCoord[orientation] - r0);
                else if(orientation == 1)
                    texCoord.y = abs(texCoord[orientation] - r0);

                vec2 cellOffset = 0;
                
                //cellOffset[orientation] = (f[orientation] > r0) ? +1. : -1.;
                if(orientation == 0)
                    cellOffset.x = (f[orientation] > r0) ? +1. : -1.;
                else if(orientation == 1)
                    cellOffset.y = (f[orientation] > r0) ? +1. : -1.;
                
                float r1 = rand(c + cellOffset);
                
                if(f[orientation ^ 1] > r1){
                    if(orientation ^ 1 == 0)
                        tileCoord.x += 1.;
                    else
                        tileCoord.y += 1.;
                }
                    
                
                //texCoord[orientation ^ 1] = abs(texCoord[orientation ^ 1] - r1);
                if(orientation ^ 1 == 0)
                    texCoord.x = abs(texCoord[orientation ^ 1] - r1);
                else if(orientation ^ 1 == 1)
                    texCoord.y = abs(texCoord[orientation ^ 1] - r1);
                
                
                // Shading and colouration.
                
                
                // Rounded box shape.
                float mask = 1. - smoothstep(0., .01, distance(vec2(.1, .1), min(texCoord, 0.1)) - .05);
                
                // Corner 'bolts' shape.
                mask *= smoothstep(.05, .06, distance(vec2(.15,.15), texCoord));
                
                // Randomised per-tile colour.
                vec3 colour = .52 + .48 * cos(Hash21(tileCoord) * vec3(4, 2, 6) * 4.5 + 18.5);
                
                // Combine.
                vec3 col = mix(vec3(.02, .02, .02), colour, mask);
                
                // Output.
                fragColor = vec4(pow(col, vec3(1. / 2.2, 1. / 2.2, 1. / 2.2)), 1.0);
                
                float cmx = 0.;
                cmx = max(cmx, fragColor.x);
                cmx = max(cmx, fragColor.y);
                cmx = max(cmx, fragColor.z);
                float dif = 1. - cmx;
                fragColor = vec4(fragColor.x + dif, fragColor.y + dif, fragColor.z + dif, 1);
                
                return fragColor * _Color1 + _Color2;
            }

            ENDCG
        }
    }
}
