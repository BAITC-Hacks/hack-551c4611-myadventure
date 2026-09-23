'use client';
import { useEffect, useState } from 'react';
import { Sun, Moon, BriefcaseBusiness, Leaf } from 'lucide-react';
type Preferences = { theme: 'light' | 'dark'; mood: 'work' | 'calm' };
const key = 'jinalys-appearance';
export default function Appearance() {
 const [prefs,setPrefs]=useState<Preferences>({theme:'light',mood:'work'});
 const [ready,setReady]=useState(false);
 const [saved,setSaved]=useState(true);
 useEffect(()=>{try{const stored=JSON.parse(localStorage.getItem(key)??'null');if(stored&&['light','dark'].includes(stored.theme)&&['work','calm'].includes(stored.mood))setPrefs(stored);}catch{setSaved(false);}setReady(true);},[]);
 useEffect(()=>{if(!ready)return;document.documentElement.dataset.theme=prefs.theme;document.documentElement.dataset.mood=prefs.mood;try{localStorage.setItem(key,JSON.stringify(prefs));setSaved(true);}catch{setSaved(false);}},[prefs,ready]);
 return <section className="appearance" aria-label="Көрініс баптаулары"><div className="appearance-intro"><span className="appearance-title">Өз ырғағыңызбен</span><span aria-live="polite">{prefs.mood==='work'?'Нақты түстер. Жинақы жұмыс кеңістігі.':'Жұмсақ реңктер. Асықпай оқуға арналған кеңістік.'}</span></div><div className="appearance-options"><div className="appearance-group" role="group" aria-label="Тақырып"><button disabled={!ready} aria-pressed={prefs.theme==='light'} onClick={()=>setPrefs({...prefs,theme:'light'})}><Sun size={15}/>Жарық</button><button disabled={!ready} aria-pressed={prefs.theme==='dark'} onClick={()=>setPrefs({...prefs,theme:'dark'})}><Moon size={15}/>Қараңғы</button></div><div className="appearance-group" role="group" aria-label="Көңіл-күй"><button disabled={!ready} aria-pressed={prefs.mood==='work'} onClick={()=>setPrefs({...prefs,mood:'work'})}><BriefcaseBusiness size={15}/>Жұмыс</button><button disabled={!ready} aria-pressed={prefs.mood==='calm'} onClick={()=>setPrefs({...prefs,mood:'calm'})}><Leaf size={15}/>Тыныш</button></div></div>{!saved&&<small className="preference-note">Баптау осы бетте қолданылады. Браузерде сақтау қолжетімсіз.</small>}</section>;
}
