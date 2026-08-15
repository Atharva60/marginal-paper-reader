export type SourceBox={page:number;x:number;y:number;width:number;height:number};
export type Passage={id:string;sectionId:string;text:string;summary:string;page:number;boxes:SourceBox[];kind:"passage"|"figure"|"table"};
export type Section={id:string;label:string;title:string;pageStart:number;pageEnd:number;text:string};
export type FigureMap={id:string;page:number;caption:string;summary:string;box:SourceBox};
export type RepoVerdict={status:"found"|"inferred"|"none";url?:string;label?:string;evidence:string};
export type TraceStep={tool:string;reason:string;at:string;result:string};
export type PaperMap={title:string;authors:string;venue:string;pageCount:number;sections:Section[];passages:Passage[];figures:FigureMap[];repo:RepoVerdict;trace:TraceStep[];mode:"gemini"|"local"};
