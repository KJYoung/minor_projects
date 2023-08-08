import React from "react";
import { CombinedTrxnGridItem, TrxnGridHeader } from "../../components/Trxn/TrxnGrid";
import { ViewMode } from "./TrxnMain";
import { CalMonth } from "../../utils/DateTime";
import { useSelector } from "react-redux";
import { selectTrxn } from "../../store/slices/trxn";
import { styled } from "styled-components";

interface TrxnCombinedProps {
    curMonth: CalMonth,
    setCurMonth?:  React.Dispatch<React.SetStateAction<CalMonth>>,
}

export const TrxnCombined = ({curMonth} : TrxnCombinedProps) => {
  const { combined }  = useSelector(selectTrxn);

  return <CombinedBodyWrapper>
    <LeftBodyWrapper>
      <TrxnGridHeader viewMode={ViewMode.Combined}/>
      {combined.map((e, index) => <CombinedTrxnGridItem index={index} date={`${curMonth.year}.${curMonth.month}.${index}`} amount={e.amount} tag={e.tag} />)}
    </LeftBodyWrapper>
    <RightBodyWrapper>

    </RightBodyWrapper>
  </CombinedBodyWrapper>;
};

const CombinedBodyWrapper = styled.div`
  width: 100%;
  height: 100%;

  display: flex;
`; 

const LeftBodyWrapper = styled.div`
  width: 50%;
  height: 100%;
  
  display: flex;
  flex-direction: column;

  border: 1px solid green;
  `;
const RightBodyWrapper = styled.div`
  width: 50%;
  height: 100%;
  
  display: flex;
  border: 1px solid green;
`;

