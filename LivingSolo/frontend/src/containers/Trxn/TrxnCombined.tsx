import React from "react";
import { CombinedTrxnGridItem, TrxnGridHeader } from "../../components/Trxn/TrxnGrid";
import { ViewMode } from "./TrxnMain";
import { CalMonth } from "../../utils/DateTime";
import { useSelector } from "react-redux";
import { selectTrxn } from "../../store/slices/trxn";

interface TrxnCombinedProps {
    curMonth: CalMonth,
    setCurMonth?:  React.Dispatch<React.SetStateAction<CalMonth>>,
}

export const TrxnCombined = ({curMonth} : TrxnCombinedProps) => {
  const { combined }  = useSelector(selectTrxn);

  return <>
      <TrxnGridHeader viewMode={ViewMode.Combined}/>
      {combined.map((e, index) => <CombinedTrxnGridItem index={index} date={`${curMonth.year}.${curMonth.month}.${index}`} amount={e} />)}
  </>;
}