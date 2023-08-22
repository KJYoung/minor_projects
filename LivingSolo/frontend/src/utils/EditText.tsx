import { styled } from "styled-components";

export const EditTextInput = styled.input`
    border: 1px solid gray;
    border-radius: 10px;
    padding: 10px 0px 10px 10px;
    background-color: var(--ls-white);

    text-align: right;
`;

export const UnderlineEditText = styled.input`
    border: none;
    border-bottom: 1px solid gray;
    padding: 4px 0px 4px 4px;

    width: 100%;

    &:active, &:focus{
        border-bottom: 1px solid var(--ls-blue);
    }
`;