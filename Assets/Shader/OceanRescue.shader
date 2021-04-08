Shader "Unlit/OceanRescue"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            #define T(a) tex2D(_MainTex,p.xz*.1-t*a)

            fixed4 frag (v2f i) : SV_Target
            {
                fixed2 uv = i.uv; uv = 2.0 * uv - 1.0;
                fixed4 fragColor;
                fixed2 iResolution = fixed2(800, 450);
                fixed2 fragCoord = i.uv * iResolution;
                fixed4 o;

                float4 p = float4(fragCoord,0.,1.)/iResolution.xyxy-.5, d=p, e;
                float t = _Time.y+6., x;
                d.y -= .2;
                p.z += t*.3;
                for(float i=1.; i>0.; i-=.02)
                {
                    e = sin(p*6.+t);
                    x = abs(p.y+e.x*e.z*.1-.75)-(e=T(.01)+T(.02)).x*.08;
                    o = .3/length(p.xy+float2(sin(t),-.4)) - e*i*i;
                    if(x<.01) break;
                    p -= d*x*.5;
                }

                return o;
            }

            ENDCG
        }
    }
}
