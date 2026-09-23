export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
const backend = 'http://127.0.0.1:8765';
export async function POST(request: Request) {
 const origin = request.headers.get('origin');
 if (origin && origin !== new URL(request.url).origin) return Response.json({error:{message:'Origin not allowed'}},{status:403});
 const size = Number(request.headers.get('content-length'));
 if (!Number.isSafeInteger(size) || size <= 0 || size > 400*1024*1024) return Response.json({error:{message:'Expected audio up to 400 MiB'}},{status:413});
 if (request.headers.get('content-type') !== 'application/octet-stream') return Response.json({error:{message:'Raw audio upload required'}},{status:415});
 const format = request.headers.get('x-audio-format')??'';
 const metadata = request.headers.get('x-meeting-metadata')??'e30=';
 if (!['wav','mp3'].includes(format) || metadata.length>2048) return Response.json({error:{message:'Invalid upload headers'}},{status:400});
 try {
  const options: RequestInit & {duplex:'half'} = {method:'POST',body:request.body,duplex:'half',signal:AbortSignal.timeout(150000),headers:{'content-type':'application/octet-stream','content-length':String(size),'x-audio-format':format,'x-meeting-metadata':metadata}};
  const r=await fetch(backend+'/jobs', options);
  return new Response(r.body,{status:r.status,headers:{'content-type':'application/json','cache-control':'no-store'}});
 } catch {return Response.json({error:{message:'Локалды серверге қосылмады. python -m integration.server іске қосыңыз.'}},{status:503});}
}
export async function GET(request: Request) {
 const id=new URL(request.url).searchParams.get('jobId');
 if(!id||!/^[a-f0-9-]{36}$/.test(id))return Response.json({error:{message:'Invalid jobId'}},{status:400});
 try{const r=await fetch(backend+'/jobs/'+id,{cache:'no-store',signal:AbortSignal.timeout(10000)});return new Response(r.body,{status:r.status,headers:{'content-type':'application/json','cache-control':'no-store'}});}
 catch{return Response.json({error:{message:'Локалды сервер қолжетімсіз.'}},{status:503});}
}
