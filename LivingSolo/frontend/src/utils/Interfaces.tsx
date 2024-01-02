export interface IPropsColor {
    color?: string;
}

export interface IPropsActive {
    // If you intentionally want it to appear in the DOM as a custom attribute, spell it as lowercase `isactive` instead.
    active?: string;
}

export interface IDReqType {
    id: number | string;
}

export interface IPropsVarGrid {
    gridlength: number;
}