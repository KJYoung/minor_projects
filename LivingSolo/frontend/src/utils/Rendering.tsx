import styled from "styled-components";

export const AutogrowInputWrapper = styled.span`
    position: relative;
`;
export const HiddenTextforAutogrowInput = styled.span`
    visibility: hidden;
    padding: 0 1rem;
`;

export type CondRendAnimState = {
    isMounted: boolean,
    showElem: boolean,
};

export const defaultCondRendAnimState: CondRendAnimState = {
    isMounted: false,
    showElem: false
};

export const condRendMounted = { animation: "inAnimation 250ms ease-in" };
export const condRendUnmounted = {
  animation: "outAnimation 250ms ease-out",
  animationFillMode: "forwards"
};

export const toggleCondRendAnimState = (curState: CondRendAnimState, setCurState: React.Dispatch<React.SetStateAction<CondRendAnimState>>) => {
    setCurState((cS) => { return { isMounted: !cS.isMounted, showElem: cS.showElem }});
    if(!curState.showElem) setCurState((cS) => { return { isMounted: cS.isMounted, showElem: true }});
};
export const onAnimEnd = (curState: CondRendAnimState, setCurState: React.Dispatch<React.SetStateAction<CondRendAnimState>>) => {
    if(!curState.isMounted) setCurState((cS) => { return { isMounted: cS.isMounted, showElem: false }});
};