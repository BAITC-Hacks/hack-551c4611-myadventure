export async function POST() {
 return Response.json({error:{code:'PIPELINE_NOT_CONFIGURED',message:'Local speech and protocol modules are not integrated yet.'}},{status:503});
}
