import type { MeetingProtocol } from './contracts.ts';
export const demoProtocol: MeetingProtocol = {
 title:'Жобаның апталық жиналысы', date:'2026-09-23',
 summary:'Команда жобаның іске қосылуына дайындықты талқылады. Қаржылық есеп пен жаңартылған жұмыс кестесін дайындау келісілді. Тестілеуге қосымша уақыт қажет; оның жауаптысы мен мерзімі әлі анықталмаған.',
 topics:[{title:'Іске қосуға дайындық',summary:'Қаржылық есеп пен мердігердің кестесін жаңарту қажет.'},{title:'Сапаны тексеру',summary:'Тестілеуді өткізу келісілді. Жауапты мен мерзімді нақтылау керек.'}],
 transcript:[
 {id:'s1',speakerId:'SPEAKER_00',speakerName:'Айгүл',start:0,end:14,text:'Әріптестер, жобаны іске қосуға дайындықты талқылайық. Асқар, қаржылық есепті жұмаға дейін дайындаңыз.',language:'kk'},
 {id:'s2',speakerId:'SPEAKER_01',speakerName:'Асқар',start:14,end:23,text:'Жақсы, есепті жұмаға дейін дайындаймын.',language:'kk'},
 {id:'s3',speakerId:'SPEAKER_00',speakerName:'Айгүл',start:23,end:39,text:'Дана, осы аптада подрядчикпен сөйлесіп, новый график жасап беріңіз.',language:'mixed'},
 {id:'s4',speakerId:'SPEAKER_02',speakerName:'Дана',start:39,end:48,text:'Хорошо, свяжусь с подрядчиком и обновлю график.',language:'ru'},
 {id:'s5',speakerId:'SPEAKER_00',speakerName:'Айгүл',start:48,end:62,text:'Іске қосар алдында қосымша тестілеу өткізу керек. Жауапты адам мен уақытын кейін белгілейміз.',language:'kk'}],
 actionItems:[
 {id:'a1',task:'Қаржылық есепті дайындау',assignee:'Асқар',deadlineText:'Жұмаға дейін',deadline:'2026-09-25',sourceSegmentIds:['s1','s2'],confidence:0.98,needsReview:false},
 {id:'a2',task:'Мердігермен сөйлесіп, жұмыс кестесін жаңарту',assignee:'Дана',deadlineText:'Осы аптада',deadline:null,sourceSegmentIds:['s3','s4'],confidence:0.91,needsReview:true},
 {id:'a3',task:'Іске қосар алдында қосымша тестілеу өткізу',assignee:null,deadlineText:null,deadline:null,sourceSegmentIds:['s5'],confidence:0.78,needsReview:true}]
};

