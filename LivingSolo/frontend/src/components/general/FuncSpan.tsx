import styled from "styled-components";
import { IPropsActive } from "../../utils/Interfaces";

interface CharNumSpanProps {
    currentNum: number,
    maxNum: number
};

export const CharNumSpan = ({ currentNum, maxNum } : CharNumSpanProps ) => {
    return <div>
        <CharNumSpanColorText active={currentNum < maxNum ? 'true' : 'false'}>{currentNum}</CharNumSpanColorText>
        /
        <span>{maxNum}</span>
    </div>
};

const CharNumSpanColorText = styled.span<IPropsActive>`
    color: ${props => ((props.active === 'true') ? 'var(--ls-blue)' : 'var(--ls-red)')};
`;