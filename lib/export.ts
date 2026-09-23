import { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType } from 'docx';
import type { MeetingProtocol } from './contracts.ts';
export async function createProtocolBlob(protocol: MeetingProtocol, demo: boolean) {
 const p = (text:string) => new Paragraph({children:[new TextRun({text,font:'Calibri',size:22})],spacing:{after:140}});
 const cell = (text:string) => new TableCell({children:[p(text)]});
 const doc = new Document({sections:[{children:[
 new Paragraph({text:protocol.title,heading:HeadingLevel.TITLE}),
 ...(demo?[p('ДЕМО ҮЛГІСІ — нақты аудио өңделген жоқ')]:[]),
 p(`Күні: ${protocol.date ?? 'Көрсетілмеген'}`),
 new Paragraph({text:'Қысқаша қорытынды',heading:HeadingLevel.HEADING_1}),p(protocol.summary),
 new Paragraph({text:'Негізгі тақырыптар',heading:HeadingLevel.HEADING_1}),...protocol.topics.map(t=>p(`${t.title}: ${t.summary}`)),
 new Paragraph({text:'Тапсырмалар',heading:HeadingLevel.HEADING_1}),
 new Table({width:{size:100,type:WidthType.PERCENTAGE},rows:[new TableRow({tableHeader:true,children:['№','Тапсырма','Жауапты','Мерзім'].map(cell)}),...protocol.actionItems.map((a,i)=>new TableRow({children:[String(i+1),a.task+(a.needsReview?' [Тексеру қажет]':''),a.assignee??'Анықталмаған',a.deadlineText??a.deadline??'Көрсетілмеген'].map(cell)}))]}),
 new Paragraph({text:'Толық мәтін',heading:HeadingLevel.HEADING_1}),...protocol.transcript.map(s=>p(`[${Math.floor(s.start/60)}:${String(Math.floor(s.start%60)).padStart(2,'0')}] ${s.speakerName??s.speakerId}: ${s.text}`))
 ]}]});
 return Packer.toBlob(doc);
}
export async function exportProtocol(protocol: MeetingProtocol, demo: boolean) {
 const blob = await createProtocolBlob(protocol,demo); const url = URL.createObjectURL(blob);
 const a = document.createElement('a'); a.href=url; a.download=demo?'jinalys-demo.docx':'jinalys-protocol.docx'; document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),60000);
}

