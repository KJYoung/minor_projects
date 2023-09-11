import { styled } from "styled-components";

export const RoundButton = styled.button`
    width: 35px;
    height: 35px;
    border: 1px solid gray;
    border-radius: 50%;
    background-color: var(--ls-blue);
    color: var(--ls-white);

    font-size: 26px;
    font-weight: 100;
    text-align: center;

    cursor: pointer;

    display: flex;
    justify-content: center;
    align-items: center;
`;

