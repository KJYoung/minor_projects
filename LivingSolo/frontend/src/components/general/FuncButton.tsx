import styled from "styled-components";
import { IPropsActive } from "../../utils/Interfaces";

interface BtnProps {
    handler: () => void,
    disabled?: boolean
};

interface EditBtnProps extends BtnProps{

};

export const EditBtn = ({ handler, disabled } : EditBtnProps ) => {
    const active = (disabled ? false : true).toString();
    return <EditBtnWrapper active={active} onClick={handler}>
        <span>수정</span>
    </EditBtnWrapper>
};

export const EditCompleteBtn = ({ handler, disabled } : EditBtnProps ) => {
    const active = (disabled ? false : true).toString();
    return <EditCompleteBtnWrapper active={active} onClick={() => !disabled && handler()}>
        <span>수정 완료</span>
    </EditCompleteBtnWrapper>
};

interface DeleteBtnProps extends BtnProps {
    confirmText?: string
};

export const DeleteBtn = ({ handler, confirmText, disabled } : DeleteBtnProps ) => {
    const active = (disabled ? false : true).toString();
    return <DeleteBtnWrapper active={active} onClick={async () => {
        if(window.confirm(confirmText ? confirmText : '정말 삭제하시겠습니까?')){
            handler();
        }
    }}>
        <span>삭제</span>
    </DeleteBtnWrapper>
};

const AbstractBtn = styled.div<IPropsActive>`
    width: 50px;
    height: 25px;
    padding: 6px 3px;
    margin-left: 3px;
    
    border-radius: 12px;
    font-size: 12px;
    text-align: center;

    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;

    cursor: ${props => ((props.active === 'true') ? 'pointer' : 'default')};
    color: ${props => ((props.active === 'true') ? 'var(--ls-black)' : 'var(--ls-gray)')};
`;

const EditBtnWrapper = styled(AbstractBtn)`
    background-color: var(--ls-blue_gray);
`;
const EditCompleteBtnWrapper = styled(AbstractBtn)`
    width: 70px;
    background-color: var(--ls-blue_gray);
`;
const DeleteBtnWrapper = styled(AbstractBtn)`
    background-color: var(--ls-red_lighter);
`;